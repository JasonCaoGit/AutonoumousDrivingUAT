# This code is based on: https://github.com/stefanknegt/Probabilistic-Unet-Pytorch
import torch
import torch.nn as nn
from torch.distributions import Normal, Independent, kl
from .utils_model import init_weights

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    """
    A convolutional neural network, consisting of len(num_filters) times a block of no_convs_per_block convolutional layers,
    after each block a pooling operation is performed. And after each convolutional layer a non-linear (ReLU) activation function is applied.
    """

    def __init__(self, input_channels, num_filters, no_convs_per_block, initializers, padding=True, posterior=False):
        super(Encoder, self).__init__()
        self.input_channels = input_channels
        self.num_filters = num_filters

        if posterior:
            # To accomodate for the mask that is concatenated at the channel axis, we increase the input_channels.
            self.input_channels += 1

        layers = []  
        for i in range(len(self.num_filters)):
            """
            Determine input_dim and output_dim of conv layers in this block. The first layer is input x output,
            All the subsequent layers are output x output.
            """
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = num_filters[i]
            if i != 0:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))
            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=int(padding)))
            layers.append(nn.ReLU(inplace=True))
            for _ in range(no_convs_per_block - 1):  
                layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=int(padding)))
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)  

        self.layers.apply(init_weights) 

    def forward(self, input):
        output = self.layers(input)
        return output


class AxisAlignedConvGaussian(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with axis aligned covariance matrix.
    """

    def __init__(self, input_channels, num_filters, no_convs_per_block, latent_dim, initializers, posterior=False):
        super(AxisAlignedConvGaussian, self).__init__()
        self.input_channels = input_channels
        self.channel_axis = 1
        self.num_filters = num_filters
        self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim  
        self.posterior = posterior  
        if self.posterior:
            self.name = 'Posterior'
        else:
            self.name = 'Prior'
        self.encoder = Encoder(self.input_channels, self.num_filters, self.no_convs_per_block, initializers,
                               posterior=self.posterior)
        self.conv_layer = nn.Conv2d(num_filters[-1], 2 * self.latent_dim, (1, 1), stride=1)
        self.show_img = 0
        self.show_seg = 0
        self.show_concat = 0
        self.show_enc = 0
        self.sum_input = 0

        nn.init.kaiming_normal_(self.conv_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.conv_layer.bias)

    def forward(self, input, segm=None):

        # If segmentation is not none, concatenate the mask to the channel axis of the input
        if segm is not None:
            self.show_img = input
            self.show_seg = segm
            input = torch.cat((input, segm), dim=1)
            self.show_concat = input
            self.sum_input = torch.sum(input)
        encoding = self.encoder(input)
        self.show_enc = encoding

        encoding = torch.mean(encoding, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)
        mu_log_sigma = self.conv_layer(encoding)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu = mu_log_sigma[:, :self.latent_dim]  
        log_sigma = mu_log_sigma[:, self.latent_dim:]  
        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)), 1)
        return dist

class Probabilistic_model(nn.Module):
    def __init__(self, input_channels, latent_dim, num_filters=[32, 64, 128, 192]):
        super(Probabilistic_model, self).__init__()
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.no_convs_per_block = 3
        self.initializers = {'w': 'he_normal', 'b': 'normal'}

        self.prior = AxisAlignedConvGaussian(self.input_channels, self.num_filters, self.no_convs_per_block,
                                             self.latent_dim, self.initializers, ).to(device)
        self.posterior = AxisAlignedConvGaussian(self.input_channels, self.num_filters, self.no_convs_per_block,
                                                 self.latent_dim, self.initializers, posterior=True).to(device)

    def forward(self, patch, segm):
        """
        Construct prior latent space for patch and run patch through UNet,
        in case training is True also construct posterior latent space
        """
        self.prior_latent_space = self.prior.forward(patch)
        self.posterior_latent_space = self.posterior.forward(patch, segm)
        if self.training:
            z_posterior = self.posterior_latent_space.rsample()
            z = z_posterior
        else:
            z_prior = self.prior_latent_space.sample()
            z = z_prior
        kl_div = self.kl_divergence()
        return z, kl_div
    def kl_divergence(self, analytic=True, calculate_posterior=False, z_posterior=None):
        """
        Calculate the KL divergence between the posterior and prior KL(Q||P)
        analytic: calculate KL analytically or via sampling from the posterior
        calculate_posterior: if we use samapling to approximate KL we can sample here or supply a sample
        """
        if analytic:  
            kl_div = kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space)
        else:  
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
            log_posterior_prob = self.posterior_latent_space.log_prob(z_posterior)  
            log_prior_prob = self.prior_latent_space.log_prob(z_posterior)  
            kl_div = log_posterior_prob - log_prior_prob  
        return kl_div
