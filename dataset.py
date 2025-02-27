import os
import random
import torch
from torch.utils.data import Dataset, Subset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class SingleObjectDataset(Dataset):
    """
    Modified dataset class for a single object per subfolder.
    Each subfolder (e.g. "car_1") contains a raw image (e.g. "car_1.png")
    and multiple segmentation masks (e.g. "car_1_seg_1.png", "car_1_seg_2.png", ...).
    """
    def __init__(self, data_path, transform=None, image_size=512, mode='Training'):
        # Keep the same folder structure: data_path/Training-400, etc.
        self.data_path = data_path
        self.mode = mode
        # Each subfolder under mode+'-400' represents one instance.
        self.subfolders = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.join(data_path, mode + '-400'))
                           for f in dn if os.path.isdir(os.path.join(dp, f))]
        self.img_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.subfolders)

    def majority_vote(self, masks):
        """
        Given a tensor of shape (N, H, W) of binary masks,
        compute the majority vote mask.
        """
        _, H, W = masks.size()
        final_mask = torch.zeros((1, H, W), dtype=torch.float32, device=masks.device)
        for h in range(H):
            for w in range(W):
                votes = torch.sum(masks[:, h, w])
                if votes > masks.size(0) // 2:
                    final_mask[0, h, w] = 1
        return final_mask

    def __getitem__(self, index):
        """
        For each subfolder, load the raw image and all segmentation mask files.
        Then choose one random mask and compute the majority vote mask.
        Return a dictionary with the processed image, random mask,
        majority vote mask, and meta data.
        """
        subfolder = self.subfolders[index]
        name = os.path.basename(subfolder)
        # Raw image filename: <name>.png (assume raw images are PNG)
        img_path = os.path.join(subfolder, name + '.png')
        # Get all segmentation mask files matching the pattern <name>_seg_*.png
        mask_files = [os.path.join(subfolder, f) for f in os.listdir(subfolder)
                      if f.startswith(name + '_seg_') and f.endswith('.png')]
        if not mask_files:
            raise ValueError(f"No segmentation masks found in {subfolder}")

        # Open raw image and resize (using bilinear interpolation)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)

        # Open each segmentation mask as grayscale ("L") and resize (using nearest)
        masks = [Image.open(path).convert('L').resize((self.img_size, self.img_size), Image.NEAREST)
                 for path in mask_files]

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            # Apply the transform to each mask then threshold (result is 1 channel)
            masks = [torch.as_tensor((self.transform(single_mask) > 0.5).float(), dtype=torch.float32)
                     for single_mask in masks]
            masks = torch.stack(masks, dim=0)  # shape: (N, 1, H, W)
            masks = torch.squeeze(masks, dim=1)  # shape: (N, H, W)
            random_mask = random.choice(masks)
            # Ensure the random mask has a channel dimension so its shape is (1, H, W)
            if random_mask.ndim == 2:
                random_mask = random_mask.unsqueeze(0)
            majority_mask = self.majority_vote(masks)
            torch.set_rng_state(state)
        else:
            img = transforms.ToTensor()(img)
            masks = [transforms.ToTensor()(single_mask) for single_mask in masks]
            masks = torch.stack(masks, dim=0)
            masks = torch.squeeze(masks, dim=1)
            random_mask = random.choice(masks)
            if random_mask.ndim == 2:
                random_mask = random_mask.unsqueeze(0)
            majority_mask = self.majority_vote(masks)

        image_meta_dict = {'filename_or_obj': name}
        return {
            'image': img,
            'random_mask_label': random_mask,
            'majorityvote_label': majority_mask,
            'all_masks': masks,
            'image_meta_dict': image_meta_dict,
        }


# The LIDC_IDRI class remains unchanged for compatibility.
class LIDC_IDRI(Dataset):
    def __init__(self, dataset_location, transform=None, split_ratio=(0.8, 0.2)):
        self.transform = transform
        self.data = []
        self.labels = []
        self.series_uid = []
        self.image_sizes = []
        self.load_data(dataset_location)
        self.train_indices, self.val_indices = self._split_dataset(len(self.data), split_ratio)

    def load_data(self, file_path):
        import pickle
        max_bytes = 2 ** 31 - 1
        bytes_in = bytearray(0)
        input_size = os.path.getsize(file_path)
        with open(file_path, 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        new_data = pickle.loads(bytes_in)
        for key, value in new_data.items():
            self.data.append(value['image'].astype(float))
            self.labels.append(value['masks'])
            self.series_uid.append(value['series_uid'])
            self.image_sizes.append(value['image'].shape)

    def _split_dataset(self, total_size, split_ratio):
        indices = list(range(total_size))
        train_end = int(split_ratio[0] * total_size)
        val_end = train_end + int(split_ratio[1] * total_size)
        return indices[:train_end], indices[train_end:val_end]

    def resize_channel_totensor(self, image):
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        image_tensor = torch.from_numpy(image).float()
        image_tensor = image_tensor.permute(2, 0, 1)
        return image_tensor

    def majority_vote(self, masks):
        num_masks, height, width = masks.shape
        final_mask = np.zeros((height, width), dtype=float)
        if np.sum(np.all(masks == 0, axis=(1, 2))) >= 2:
            return final_mask
        for h in range(height):
            for w in range(width):
                votes = np.sum(masks[:, h, w])
                if votes > num_masks // 2:
                    final_mask[h, w] = 1
        return final_mask

    def __getitem__(self, index):
        image = self.data[index]
        masks_four = np.array(self.labels[index])
        label = self.majority_vote(masks_four).astype(float)
        masks = self.labels[index]
        transform_label = transforms.Compose([transforms.ToTensor()])
        if isinstance(label, np.ndarray):
            label = Image.fromarray((label * 255).astype(np.uint8))
        if isinstance(masks, list):
            masks = [Image.fromarray((mask * 255).astype(np.uint8)) for mask in masks]
        if self.transform:
            image = self.resize_channel_totensor(image)
            label = transform_label(label)
            masks = [transform_label(mask) for mask in masks]
        threshold = 0.5
        label = (label > threshold).float()
        masks = [(mask > threshold).float() for mask in masks]
        return {
            'image': image,
            'majorityvote_label': label,
            'random_label': random.choice(masks),
            'all_masks': torch.stack(masks),
        }

    def __len__(self):
        return len(self.data)

    def get_train_val_test(self):
        from torch.utils.data import Subset
        train_dataset = Subset(self, self.train_indices)
        val_dataset = Subset(self, self.val_indices)
        return train_dataset, val_dataset


def build_dataset(args):
    transform_refuge = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])
    transform_lidc = transforms.Compose([
        transforms.ToTensor(),
    ])
    if args.dataset == 'refuge':
        # Use our modified SingleObjectDataset for the refuge dataset.
        train_dataset = SingleObjectDataset(args.dataset_path, transform=transform_refuge, mode='Training')
        val_dataset = SingleObjectDataset(args.dataset_path, transform=transform_refuge, mode='Validation')
        test_dataset = SingleObjectDataset(args.dataset_path, transform=transform_refuge, mode='Test')
    elif args.dataset == 'lidc':
        full_dataset = LIDC_IDRI(args.dataset_path, transform=transform_lidc, split_ratio=args.split_ratio)
        train_dataset, val_dataset = full_dataset.get_train_val_test()
        test_dataset = val_dataset
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    return train_dataset, val_dataset, test_dataset
