import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from loss import *
from script.eval import *
from script.utils import *
from PIL import Image
from tqdm import tqdm
from torchsummary import summary
from torch.nn import Conv2d
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR
from torchvision import transforms

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
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
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

def train(model, config, train_loader, valid_loader, criterion, optimizer, scheduler=None):
    # should be in conofig
    max_epochs = 30
    
    # training
    model.train()
    for epoch in range(max_epochs):
        total_loss = 0.
        iou_meter = AverageMeter()
        iou_meter_sequence = AverageMeter()
        label_validity = []
        output_conf = []
        for batch, data in enumerate(tqdm(train_loader, leave=True)):
            # [batch, 3, h, w], [batch, 1, h, w]
            images, masks = data['images'].to(device), data['masks'].to(device)
            # print(images.dtype, masks.dtype)
            preds = model(images) # [batch, 1, h, w]
            loss = criterion(preds, masks)
            # print(preds.shape)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Evaluate
            total_loss += loss.item()
            '''from reference???'''
            masks = masks.cpu().detach().numpy()
            preds = preds.cpu().detach().numpy()

            for batch in range(preds.shape[0]):
                output = preds[batch]
                label = masks[batch]

                if np.sum(output.flatten()) > 0:
                    conf = 1.0
                    iou = mask_iou(output, label)
                    iou_meter.update(conf * iou)
                    iou_meter_sequence.update(conf * iou)
                else:  # empty ground truth label
                    conf = 0.0

                if np.sum(label.flatten()) > 0:
                    label_validity.append(1.0)
                else:  # empty ground truth label
                    label_validity.append(0.0)
                    
                output_conf.append(conf)
        
                pass
    # print each epoch's result

        wiou = iou_meter.avg()
        tn_rates = true_negative_curve(np.array(output_conf), np.array(label_validity))
        atnr = np.mean(tn_rates)
        score = 0.7 * wiou + 0.3 * atnr
        
    # validation
    model.eval()
    for batch, data in enumerate(tqdm(valid_loader)):
        with torch.no_grad():
            images, masks = data['images'].to(device), data['masks'].to(device)
            
            preds = model(images)
            loss = criterion(preds, masks)

            total_loss += loss.item()
    pass

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    
    train_loader, valid_loader = get_dataloader('./dataset', batch_size=16, split='train', valid_ratio=0.1)
    # for data in trainset:
        # print(type(data['images']), type(data['masks']))
        # print(data['images'].shape, data['masks'].shape)
        # print(len(data['masks'].split()))
    
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)
    # model.conv = Conv2d(32, 4, kernel_size=(1, 1), stride=(1, 1))
    model = nn.DataParallel(model, device_ids=[0, 1])
    model.to(device)
    # summary(model, input_size=(3, 480, 640))
    # print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = DiceLoss()
    # nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=30)
    
    train(model=model,
          config='to be added',
          train_loader=train_loader,
          valid_loader=valid_loader,
          criterion=criterion,
          optimizer=optimizer,
        #   scheduler=scheduler
          )


    '''
    model.eval()
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