import numpy as np
import torch
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import Conv2d
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

def get_train_path(path='./dataset'):
    data_path = path
    image_list = []
    mask_list = []
    print('Acquiring training image & mask path...')
    for serie in tqdm(os.listdir(data_path)):
        if serie in ['S1', 'S2', 'S3', 'S4']:
            for sequence in os.listdir(f'{data_path}/{serie}'):
                seq_image = []
                seq_mask = []
                for name in os.listdir(f'{data_path}/{serie}/{sequence}'):
                    ext = os.path.splitext(name)[-1]
                    fullpath = f'{data_path}/{serie}/{sequence}/{name}'
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
            transforms.ToTensor(),
            # TODO: add other augmentations
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    dataset = GanzinDataset(image_path, mask_path, transform)
    if split == 'train':
        num_valid = round(len(dataset) * valid_ratio)
        num_train = len(dataset) - num_valid
        train_set, valid_set = random_split(dataset, [num_train, num_valid])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
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
        mask = Image.open(self.mask_paths[idx]).convert('L')

        image = self.transforms(image)
        mask = self.transforms(mask)

        return {
            'images': image,
            'masks': mask
        }

def train(model, config, loss, optimizer, train_loader, valid_loader, scheduler):
    # should be in conofig
    max_epochs = 30
    train_loss_list, val_loss_list = [], []
    train_acc_list, val_acc_list = [], []
    pass

if __name__ == '__main__':
    train_loader, valid_loader = get_dataloader('./dataset', batch_size=32, split='train', valid_ratio=0.1)
    # for data in trainset:
        # print(type(data['images']), type(data['masks']))
        # print(data['images'].shape, data['masks'].shape)
        # print(len(data['masks'].split()))
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)
    model.conv = Conv2d(32, 4, kernel_size=(1, 1), stride=(1, 1))
    # print(model)


    model.eval()
    '''
    input_image = Image.open('./dataset/S1/01/0.jpg').convert('RGB')

    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(input_image)
    print(input_tensor.shape)
    input_batch = input_tensor.unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_batch)
        print(output.shape)
    out = (output.numpy() * 255).astype(np.uint8)[0]
    print(out.shape)
    im = Image.fromarray(out.transpose((1, 2, 0)))
    im.save('./ans.png')
    '''