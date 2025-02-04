import os
import pickle
import random
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np


class REFUGE(Dataset):
    def __init__(self, data_path, transform=None, image_size=512,  mode='Training'):
        self.data_path = data_path
        self.subfolders = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.join(data_path, mode + '-400')) for f in dn]
        self.mode = mode
        self.img_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.subfolders)

    def majority_vote(self, masks):
        _, H, W = masks.size()
        final_mask = torch.zeros((1, H, W), dtype=torch.float32, device=masks.device)

        for h in range(H):
            for w in range(W):
                votes = torch.sum(masks[:, h, w])
                if votes > masks.size(0) // 2: 
                    final_mask[0, h, w] = 1 
        return final_mask

    def load_disc_info(self, xlsx_path):
        xl = pd.ExcelFile(xlsx_path)
        df = xl.parse(xl.sheet_names[0])
        disc_info_json = df.iloc[2, 2]  # Rows and columns are zero-indexed
        disc_info = json.loads(disc_info_json)
        return disc_info

    def crop_to_disc(self, image, disc_info, output_size=(384, 384)):
        """
        Crop the image around the disc center to the specified output size.
        """
        center_x, center_y = disc_info['centerX'], disc_info['centerY']
        width, height = output_size
        left = max(center_x - width // 2, 0)
        top = max(center_y - height // 2, 0)
        right = min(center_x + width // 2, image.width)
        bottom = min(center_y + height // 2, image.height)
        image = image.crop((left, top, right, bottom))
        return image

    def __getitem__(self, index):

        """Get the images"""
        subfolder = self.subfolders[index]
        name = os.path.basename(subfolder)
        img_path = os.path.join(subfolder, name + '.jpg')
        xlsx_path = os.path.join(subfolder, name + '.xlsx')
        disc_info = self.load_disc_info(xlsx_path)
        multi_rater_cup_path = [os.path.join(subfolder, name + '_seg_cup_' + str(i) + '.png') for i in range(1, 8)]
        multi_rater_disc_path = [os.path.join(subfolder, name + '_seg_disc_' + str(i) + '.png') for i in range(1, 8)]

        # raw image and raters images
        img = Image.open(img_path).convert('RGB')
        img = self.crop_to_disc(img, disc_info, output_size=(384, 384))
        multi_rater_cup = [self.crop_to_disc(Image.open(path).convert('L'), disc_info, output_size=(384, 384)) for path in multi_rater_cup_path]
        multi_rater_disc = [self.crop_to_disc(Image.open(path).convert('L'), disc_info, output_size=(384, 384)) for path in multi_rater_disc_path]

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            multi_rater_cup = [torch.as_tensor((self.transform(single_rater) > 0.5).float(), dtype=torch.float32) for single_rater in multi_rater_cup]
            multi_rater_cup = torch.stack(multi_rater_cup, dim=0)
            random_cup_label = random.choice(multi_rater_cup)  
            multi_rater_cup = torch.squeeze(multi_rater_cup, dim=1)
            majorityvote_cup_label = self.majority_vote(multi_rater_cup)  


            multi_rater_disc = [torch.as_tensor((self.transform(single_rater) > 0.5).float(), dtype=torch.float32) for single_rater in multi_rater_disc]
            multi_rater_disc = torch.stack(multi_rater_disc, dim=0)
            random_disc_label = random.choice(multi_rater_disc)  
            multi_rater_disc = torch.squeeze(multi_rater_disc, dim=1)
            majorityvote_disc_label = self.majority_vote(multi_rater_disc)  
            torch.set_rng_state(state)
        image_meta_dict = {'filename_or_obj': name}

        return {
            'image': img,
            'random_cup_label': random_cup_label,
            'all_masks': multi_rater_cup,
            'random_disc_label': random_disc_label,
            'majorityvote_cup_label': majorityvote_cup_label,
            'majorityvote_disc_label': majorityvote_disc_label,
            'multi_rater_cup': multi_rater_cup,
            'image_meta_dict': image_meta_dict,
        }

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

        transform_label = transforms.Compose([
            transforms.ToTensor(),
        ])
        
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
        train_dataset = REFUGE(args.dataset_path, transform=transform_refuge, mode='Training')
        val_dataset = REFUGE(args.dataset_path, transform=transform_refuge, mode='Validation')
        test_dataset = REFUGE(args.dataset_path, transform=transform_refuge, mode='Test')
        
    elif args.dataset == 'lidc':
        full_dataset = LIDC_IDRI(args.dataset_path, transform=transform_lidc, 
                                split_ratio=args.split_ratio)
        train_dataset, val_dataset = full_dataset.get_train_val_test()
        test_dataset = val_dataset  
    
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    return train_dataset, val_dataset, test_dataset