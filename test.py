import os
import gc
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from loss import *
from eval import *
from utils import *
from config import args
from PIL import Image
from time import time
from tqdm import tqdm
from torchsummary import summary
from torch.nn import Conv2d
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR
from torchvision import transforms

def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_train_path(path='./dataset'):
    data_path = path
    image_list = []
    mask_list = []
    print('Acquiring training image & mask path...')
    for subject in tqdm(sorted(os.listdir(data_path))):
        if subject in ['S1', 'S2', 'S3', 'S4']:
            for sequence in sorted(os.listdir(f'{data_path}/{subject}')):

                seq_image = []
                seq_mask = []
                for name in sorted(os.listdir(f'{data_path}/{subject}/{sequence}')):
                    ext = os.path.splitext(name)[-1]
                    fullpath = f'{data_path}/{subject}/{sequence}/{name}'
                    if ext == '.jpg':
                        image_list.append(fullpath)
                    elif ext == '.png':
                        mask_list.append(fullpath)
                # image_list.append(seq_image)
                # mask_list.append(seq_mask)
    print(f'Number of image:{len(image_list)}, mask:{len(mask_list)}')
    return np.array(image_list), np.array(mask_list)

def get_dataloader(data_dir, batch_size, split='test', valid_ratio=0.1):
    image_path, mask_path = get_train_path(data_dir)
    if split == 'train':
        transform = transforms.Compose([
            transforms.Resize((240, 320)),
            transforms.ToTensor(),
            # TODO: add other augmentations
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((240, 320)),
            transforms.ToTensor(),
        ])
    
    dataset = GanzinDataset(image_path, mask_path, transform)
    if split == 'train':
        num_valid = round(len(dataset) * valid_ratio)
        num_train = len(dataset) - num_valid
        train_set, valid_set = random_split(dataset, [num_train, num_valid])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        return train_loader, valid_loader
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
        return dataloader


class GanzinDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transform

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
        # print(mask.size)
        image = self.transforms(image)
        mask = self.transforms(mask)

        return {
            'images': image,
            'masks': mask
        }

if __name__ == '__main__':
    '''Choose Device'''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    
    '''Fix Random Seed'''
    set_seed(2023)
    '''Get DataLoader'''
    test_loader = get_dataloader(args.datapath, batch_size=args.batch_size, split='test', valid_ratio=0.1)

    '''Pretrained Model Selection'''
    # https://github.com/mateuszbuda/brain-segmentation-pytorch
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)
    
    '''Model Setting & Summary'''
    model = nn.DataParallel(model, device_ids=[0, 1])
    model.to(device)
    # summary(model, input_size=(3, 240, 320))
    # print(model)
    '''Read Parameters (.pth)'''
    model.load_state_dict(torch.load('./test/model_best_16.pth'))
    
    '''Selecting optimizer ...'''
    criterion = DiceLoss()

    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for batch, data in enumerate(tqdm(test_loader)):
            images, masks = data['images'].to(device), data['masks'].to(device)
            
            preds = model(images)
            loss = criterion(preds, masks)

            test_loss += loss.item()
            
            for idx in range(preds.shape[0]):
                # save visualize
                image_tmp = images[idx].cpu().detach().numpy().transpose(1, 2, 0)
                mask_tmp = masks[idx].cpu().detach().numpy().transpose(1, 2, 0)
                mask_rgb = np.zeros(image_tmp.shape)
                mask_rgb[:, :, 0] = mask_tmp[:, :, 0]
                mask_rgb[:, :, 2] = mask_tmp[:, :, 0]
                blend_tmp = alpha_blend(image_tmp*255, mask_rgb*255, 0.5)
                pred_tmp = preds[idx].cpu().detach().numpy().transpose(1, 2, 0)
                
                fig = plt.figure()

                # plt.subplot(1, 3, 1)
                # plt.imshow(image_tmp)
                # plt.subplot(1, 3, 2)
                # plt.imshow(mask_tmp)
                # plt.subplot(1, 3, 3)
                # plt.imshow(pred_tmp)
                plt.subplot(1, 2, 1)
                plt.imshow(blend_tmp)
                plt.subplot(1, 2, 2)
                plt.imshow(pred_tmp)

                # plt.show()
                fig_folder = './output'
                os.makedirs(fig_folder, exist_ok=True)
                fig.savefig(os.path.join(fig_folder, f'{batch}_{idx}.jpg'))
                plt.close()
        test_loss = test_loss / len(test_loader)

    print(f'Test Loss:{test_loss:.5f}')