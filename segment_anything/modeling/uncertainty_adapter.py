import torch
import torch.nn as nn
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class PCAtten(nn.Module):
    def __init__(self, batchsize, latent_dim, output_dim):
        super(PCAtten, self).__init__()
        self.batchsize = batchsize
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.MLP1 = nn.Sequential(
            nn.Linear(12, batchsize * latent_dim),
            nn.GELU(),
            nn.Linear(batchsize * latent_dim, batchsize * latent_dim),
            nn.GELU()
        )
        self.MLP2 = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, self.output_dim),
            nn.GELU()
        )
        self.MLP3 = nn.Sequential(
            nn.Linear(latent_dim, 2*latent_dim),
            nn.GELU(),
            nn.Linear(2*latent_dim, 12 // batchsize),
            nn.GELU()
        )
        
        self.W11 = nn.Linear(latent_dim, latent_dim)
        self.W12 = nn.Linear(latent_dim, latent_dim)
        self.W21 = nn.Linear(latent_dim, latent_dim)
        self.W22 = nn.Linear(latent_dim, latent_dim)
        self.act = nn.Softmax()
        self.act2 = nn.GELU()
        
    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(
            device)
        return torch.index_select(a, dim, order_index)
    def reconstruct(self, z):
        z = torch.unsqueeze(z, 1)
        z = self.tile(z, 1, 64)
        z = torch.unsqueeze(z, 2)
        z = self.tile(z, 2, 64)
        z = self.MLP2(z)
        return z
    def forward(self, condition, sample_z):
        p = self.MLP1(condition)
        p = p.reshape(self.batchsize, self.latent_dim)
        p_k = self.W11(p)
        p_k = p_k.transpose(0, 1)
        p_v = self.act2(self.W12(p))
        z_q = self.W21(sample_z)
        z_v = self.W22(sample_z)
        z1 = self.act(torch.matmul(p_k, z_q) / torch.sqrt(torch.tensor(self.latent_dim, dtype=torch.float32)))
        z1 = torch.matmul(z_v, z1)
        # print(f"z': {z1}")
        x = torch.mul(z1, p_v)
        g = self.reconstruct(x)
        next_p = self.MLP3(p_v)
        next_p = next_p.reshape(1, 12)
        return next_p, g


class UncertaintyAdapter(nn.Module):
    def __init__(self, D_features, batchsize, latent_dim, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super(UncertaintyAdapter, self).__init__()
        self.latent_dim = latent_dim
        self.skip_connect = skip_connect
        self.act = act_layer()
        D_hidden_features = int(D_features * mlp_ratio)
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.D_prompt = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 192),
            nn.GELU(),
            nn.Linear(192, 192),
            nn.GELU(),
        )
        self.PCAtten = PCAtten(batchsize, self.latent_dim, D_hidden_features)
    def forward(self, x, condition, sample_z, prompt_embeddings):
        xs = self.D_fc1(x)
       
        xs = self.act(xs)
        next_p, g = self.PCAtten(condition, sample_z)
        pp = self.D_prompt(prompt_embeddings)
        pp_expanded = pp.unsqueeze(2).unsqueeze(3)  # (2, 2, 1, 1, 192)
        g_expanded = g.unsqueeze(1)  # (2, 1, 64, 64, 192)
        result = torch.mul(pp_expanded, g_expanded)  # (2, 2, 64, 64, 192)
        g = result.sum(dim=1)
        xs += g
        
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x, next_p
