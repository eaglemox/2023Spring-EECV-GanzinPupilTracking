import os
import cv2
import torch
import numpy as np
import albumentations as A
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from utils import mask2dist
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split

def get_train_path(path='./dataset'):
    data_path = path
    image_list = []
    mask_list = []
    print('Acquiring training image & mask path...')
    for subject in tqdm(sorted(os.listdir(data_path))):
        if subject in ['S1', 'S2', 'S3', 'S4']:
            for sequence in sorted(os.listdir(f'{data_path}/{subject}')):
                for name in sorted(os.listdir(f'{data_path}/{subject}/{sequence}')):
                    ext = os.path.splitext(name)[-1]
                    fullpath = f'{data_path}/{subject}/{sequence}/{name}'
                    if ext == '.jpg':
                        image_list.append(fullpath)
                    elif ext == '.png':
                        mask_list.append(fullpath)

    print(f'Number of image:{len(image_list)}, mask:{len(mask_list)}')
    return np.array(image_list), np.array(mask_list)

def get_sequence_path(path='./dataset'):
    data_path = path
    image_list = []
    seq_length = len([seq for seq in os.listdir(data_path)])
    for seq in range(seq_length):
        sequence_path = f'{data_path}/{seq + 1:02d}'
        img_length = len([name for name in os.listdir(sequence_path) if name.endswith('.jpg')])
        for index in range(img_length):
            fullpath = f'{sequence_path}/{index}.jpg'
            image_list.append(fullpath)

    print(f'Number of image:{len(image_list)}')
    return np.array(image_list)

def get_dataloader(data_dir, batch_size, split='test', valid_ratio=0.1):
    image_path, mask_path = get_train_path(data_dir)
    if split == 'train':
        transform = A.Compose([
            A.Resize(240, 320),
            A.Rotate(limit=20, p=0.3, border_mode=cv2.BORDER_CONSTANT),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.Perspective(scale=(0.05, 0.2), keep_size=True, fit_output=False),
            A.RandomGamma(gamma_limit=(40, 60), p=1.0), # 40 mean gamma = 40 / 100 = 0.4
            # A.Equalize(p=1),
            A.Normalize(mean=0.5, std=0.5),
            # A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.3, border_mode=cv2.BORDER_CONSTANT),
            # A.HueSaturationValue(p=0.4),
            # # A.Sharpen(p=0.3),
            # A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.3),
            # ToTensorV2(),
            ])
                                
        img_transform = transforms.Compose([
            transforms.Resize((240, 320)),
            transforms.RandomEqualize(p=1.0),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.3, contrast= 0.5),
            transforms.ToTensor(),
            # TODO: add other augmentations
        ])
        mask_transform = transforms.Compose([
            transforms.Resize((240, 320)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ToTensor(),
            # TODO: add other augmentations
        ])
    else:
        img_transform = transforms.Compose([
            transforms.Resize((240, 320)),
            transforms.ToTensor(),
        ])
        mask_transform = transforms.Compose([
            transforms.Resize((240, 320)),
            transforms.ToTensor(),
        ])    
    dataset = GanzinDataset(image_path, mask_path, transform, img_transform, mask_transform)

    if split == 'train':
        num_valid = round(len(dataset) * valid_ratio)
        num_train = len(dataset) - num_valid
        train_set, valid_set = random_split(dataset, [num_train, num_valid])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        return train_loader, valid_loader
    else:
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
        return dataloader

def get_inference_dataloader(data_dir, batch_size):
    image_paths = get_sequence_path(data_dir)

    transform = transforms.Compose([
        transforms.Resize((240 ,320)),
        transforms.ToTensor(),
    ])
    dataset = InferenceGanzinDataset(image_paths, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    return dataloader

class GanzinDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform, img_transform, mask_transform):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.img_transforms = img_transform
        self.mask_transforms = mask_transform

    def __len__(self):
        if len(self.image_paths) == len(self.mask_paths):
            return len(self.image_paths)
        else:
            print('Number of images and mask does not match!')
            return np.min(len(self.image_paths), len(self.mask_paths))
        
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask_temp = Image.open(self.mask_paths[idx])

        mask = mask_temp.split()[0] # origin pupil mask is [255, 0, 255]-> magenta

        # torch return
        # image = self.img_transforms(image)
        # mask = self.mask_transforms(mask)
        totensor = transforms.Compose([transforms.ToTensor()])
        transformed = self.transform(image=np.array(image), mask=np.array(mask))
        image, mask = totensor(transformed['image']), totensor(transformed['mask'])

        # mask = F.one_hot(mask[0].long(), num_classes=2).permute(2, 0, 1)
        # print(mask.shape)
        # print(mask.shape)
        # image, mask = trans(transformed['image']), transformed['mask']
        
        dist_map = torch.Tensor(mask2dist(mask))
        # mask = trans(mask)
        
        return {
            'images': image,
            'masks': mask.float(),
            'dists': dist_map
        }

class InferenceGanzinDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.img_transforms = transform

    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')

        image = self.img_transforms(image)

        return {
            'images': image,
            'paths': self.image_paths[idx]
        }