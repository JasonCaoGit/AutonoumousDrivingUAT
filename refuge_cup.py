from torch.utils.data import Dataset
import os
import random
import json
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image


class REFUGE(Dataset):
    def __init__(self, data_path, transform=None, image_size=512, mode='Training'):
        self.data_path = data_path
        self.subfolders = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.join(data_path, mode + '-400')) for f in dn]
        self.mode = mode
        self.img_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.subfolders)

    def load_disc_info(self, xlsx_path):
        xl = pd.ExcelFile(xlsx_path)
        df = xl.parse(xl.sheet_names[0])
        disc_info_json = df.iloc[2, 2]
        disc_info = json.loads(disc_info_json)
        return disc_info

    def crop_to_disc(self, image, disc_info, output_size=(384, 384)):
        center_x, center_y = disc_info['centerX'], disc_info['centerY']
        width, height = output_size

        left = max(center_x - width // 2, 0)
        top = max(center_y - height // 2, 0)
        right = min(center_x + width // 2, image.width)
        bottom = min(center_y + height // 2, image.height)

        image = image.crop((left, top, right, bottom))
        return image

    def majority_vote(self, masks):
        k = masks.size(0)
        sum_masks = torch.sum(masks, dim=0)  
        threshold = k / 2.0
        aggregated_mask = (sum_masks >= threshold).float()
        return aggregated_mask

    def __getitem__(self, index):
        subfolder = self.subfolders[index]
        name = os.path.basename(subfolder)
        img_path = os.path.join(subfolder, name + '.jpg')
        xlsx_path = os.path.join(subfolder, name + '.xlsx')
        disc_info = self.load_disc_info(xlsx_path)
        multi_rater_cup_path = [os.path.join(subfolder, name + '_seg_cup_' + str(i) + '.png') for i in range(1, 8)]

        img = Image.open(img_path).convert('RGB')
        img = self.crop_to_disc(img, disc_info, output_size=(384, 384))
        multi_rater_cup = [self.crop_to_disc(Image.open(path).convert('L'), disc_info, output_size=(384, 384)) for path in multi_rater_cup_path]

        if self.transform:
            img = self.transform(img)
            multi_rater_cup = [torch.as_tensor((self.transform(single_rater) > 0.5).float(), dtype=torch.float32) for single_rater in multi_rater_cup]
            multi_rater_cup = torch.stack(multi_rater_cup, dim=0)  # [7, 1, H, W]
            multi_rater_cup = torch.squeeze(multi_rater_cup, dim=1)  # [7, H, W]

            individual_cup_masks = [multi_rater_cup[i, :, :] for i in range(7)]  

            aggregated_cup_masks = []
            for k in range(1, 8):  # 从1到7
                selected_doctors = random.sample(range(7), k)
                masks_k = multi_rater_cup[selected_doctors, :, :]  # [k, H, W]
                aggregated_mask = self.majority_vote(masks_k)  # [H, W]
                aggregated_cup_masks.append(aggregated_mask)

        image_meta_dict = {'filename_or_obj': name}

        return {
            'image': img,
            'individual_cup_masks': individual_cup_masks,  
            'aggregated_cup_masks': aggregated_cup_masks,    
            'image_meta_dict': image_meta_dict,
        }
