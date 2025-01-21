import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from monai.losses import DiceCELoss
from skimage.measure import label, regionprops
diceceloss = DiceCELoss(sigmoid=True)

def generate_click_prompt(msk, num_points=1, pt_label=1):
    """
    Generate point prompts from a mask
    Args:
        msk: Binary mask tensor of shape (B, 1, H, W)
        num_points: Number of points to generate per mask
        pt_label: Label value for the points
    Returns:
        Dictionary containing point coordinates and labels
    """
    pt_coords_list = []
    pt_labels_list = []

    b, c, h, w = msk.size()

    for j in range(b):
        msk_s = msk[j, 0, :, :]
        indices = torch.nonzero(msk_s)

        for _ in range(num_points):
            if indices.nelement() == 0:
                # If no non-zero elements, generate a random point
                random_index = torch.randint(0, h, (2,)).to(device=msk.device)
                pt_label_default = torch.tensor([pt_label], device=msk.device).view(1)
            else:
                # Sample from non-zero elements
                random_idx = torch.randint(0, indices.shape[0], (1,))
                random_index = indices[random_idx].squeeze()
                pt_label_default = msk_s[random_index[0], random_index[1]].view(1)

            pt_coords_list.append(random_index)
            pt_labels_list.append(pt_label_default)

    pt_coords = torch.stack(pt_coords_list).view(b, num_points, 2)
    pt_labels = torch.stack(pt_labels_list).view(b, num_points)

    return {
        'point_coords': pt_coords,
        'point_labels': pt_labels
    }

def random_box(mask, box_num=1, std=0.1, max_pixel=5):
    """
    Args:
        mask: Mask, should be a torch.Tensor of shape (B, 1, H, W).
        box_num: Number of bounding boxes, default is 1.
        std: Standard deviation of the noise, default is 0.1.
        max_pixel: Maximum noise pixel value, default is 5.
    Returns:
        noise_boxes: Bounding boxes after noise perturbation, returned as a torch.Tensor of shape (B, box_num, 4).
    """
    B, C, H, W = mask.shape
    noise_boxes = []

    for i in range(B):
        single_mask = mask[i, 0, :, :].cpu().numpy()  # 将张量从GPU移到CPU并转换为numpy数组

        label_img = label(single_mask)
        regions = regionprops(label_img)

        # Iterate through all regions and get the bounding box coordinates
        boxes = [tuple(region.bbox) for region in regions]

        # 如果 boxes 为空，跳过这个样本或返回默认的 box
        if len(boxes) == 0:
            noise_boxes.append([(0, 0, 0, 0) for _ in range(box_num)])
            continue

        # If the generated number of boxes is greater than the number of categories,
        # sort them by region area and select the top n regions
        if len(boxes) >= box_num:
            sorted_regions = sorted(regions, key=lambda x: x.area, reverse=True)[:box_num]
            boxes = [tuple(region.bbox) for region in sorted_regions]

        # If the generated number of boxes is less than the number of categories,
        # duplicate the existing boxes
        elif len(boxes) < box_num:
            num_duplicates = box_num - len(boxes)
            boxes += [boxes[i % len(boxes)] for i in range(num_duplicates)]

        # Perturb each bounding box with noise
        batch_noise_boxes = []
        for box in boxes:
            y0, x0, y1, x1 = box
            width, height = abs(x1 - x0), abs(y1 - y0)
            # Calculate the standard deviation and maximum noise value
            noise_std = min(width, height) * std
            max_noise = max(1, min(max_pixel, int(noise_std * 5)))  # 确保 max_noise 至少为1
            # Add random noise to each coordinate
            noise_x = np.random.randint(-max_noise, max_noise)
            noise_y = np.random.randint(-max_noise, max_noise)
            x0, y0 = x0 + noise_x, y0 + noise_y
            x1, y1 = x1 + noise_x, y1 + noise_y
            batch_noise_boxes.append((x0, y0, x1, y1))
        
        noise_boxes.append(batch_noise_boxes)

    return torch.as_tensor(noise_boxes, dtype=torch.float)


def elbo(segm, label, kl_divergence, beta):
    """
    Compute ELBO loss using DiceCE loss for reconstruction
    """
    if not isinstance(kl_divergence, torch.Tensor):
         kl_divergence = torch.tensor(kl_divergence).to(label.device)
    reconstruction_loss = diceceloss(input=segm, target=label)
    kl_loss = torch.mean(kl_divergence)
    return reconstruction_loss + beta * kl_loss

def truncated_normal_(tensor, mean=0, std=1):
    """
    Initialize tensor with truncated normal distribution
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

def init_weights(m):
    """
    Initialize network weights using Kaiming initialization
    """
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        truncated_normal_(m.bias, mean=0, std=0.001)

def init_weights_orthogonal_normal(m):
    """
    Initialize network weights using orthogonal initialization
    """
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)

def l2_regularisation(m):
    """
    Compute L2 regularization for model parameters
    """
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg

def save_mask_prediction_example(mask, pred, iter):
    """
    Save mask and prediction visualizations
    """
    plt.imshow(pred[0,:,:], cmap='Greys')
    plt.savefig('images/'+str(iter)+"_prediction.png")
    plt.close()
    plt.imshow(mask[0,:,:], cmap='Greys')
    plt.savefig('images/'+str(iter)+"_mask.png")
    plt.close()

def prepare_image(image, target_size):
    """
    Prepare image for model input
    """
    if image.dtype == torch.uint8:
        image = image.float() / 255.0
    
    if len(image.shape) == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif len(image.shape) == 3:
        image = image.unsqueeze(0)
    
    if image.shape[-2:] != target_size:
        image = F.interpolate(
            image,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )
    
    return image

def post_process_masks(masks, threshold=0.5, min_area=100):
    """
    Post-process predicted masks
    """
    binary_masks = (masks > threshold).float()
    
    for i in range(binary_masks.shape[0]):
        mask = binary_masks[i, 0].cpu().numpy()
        from scipy import ndimage
        labeled, num_features = ndimage.label(mask)
        
        for j in range(1, num_features + 1):
            component = (labeled == j)
            if component.sum() < min_area:
                mask[component] = 0
                
        binary_masks[i, 0] = torch.from_numpy(mask).to(masks.device)
    
    return binary_masks

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count