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


def get_train_path(path='./dataset'):
    data_path = path
    image_list = []
    mask_list = []
    print('Acquiring training image & mask path...')
    for subject in tqdm(os.listdir(data_path)):
        if subject in ['S1', 'S2', 'S3', 'S4']:
            for sequence in os.listdir(f'{data_path}/{subject}'):
                seq_image = []
                seq_mask = []
                for name in os.listdir(f'{data_path}/{subject}/{sequence}'):
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
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
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

        image = self.transforms(image)
        mask = self.transforms(mask)

        return {
            'images': image,
            'masks': mask
        }

def cal_batch_metric(predict_batch, label_batch, iou_list, conf_list, validframe_list):
    predict_batch = predict_batch.cpu().detach().numpy()
    label_batch = label_batch.cpu().detach().numpy()

    for batch in range(predict_batch.shape[0]):
        predict = predict_batch[batch]
        label = label_batch[batch]

        if np.sum(predict) > 0:
            conf = 1.0
            iou = mask_iou(predict, label)
            iou_list.append(conf * iou)
        else:  # empty ground truth label
            conf = 0.0

        if np.sum(label) > 0:
            validframe_list.append(1.0)
        else:  # empty ground truth label
            validframe_list.append(0.0)
            
        conf_list.append(conf)

def write_txtlog(log_path, current_epoch, train_score, valid_score, train_loss, valid_loss, train_iou, valid_iou, train_atnr, valid_atnr, is_better):
    with open(log_path, 'a') as f:
        f.write(f'[{current_epoch+1}/{args.max_epoch}] Score:{train_score}/{valid_score} | Loss:{train_loss}/{valid_loss}\
                 | IOU:{train_iou}/{valid_iou} | ATNR:{train_atnr}/{valid_atnr}')
        if is_better:
            f.wrtie('--> Best Updated')
        f.write('\n')


def train(model, train_loader, valid_loader, criterion, optimizer, scheduler=None):
    # list for learning curve
    train_loss_list, valid_loss_list = [], []
    train_score_list, valid_score_list = [], []

    # initialize best loss
    best_loss = np.inf

    # training
    model.train()
    for epoch in range(args.max_epoch):
        print(f'Epoch: {epoch}/{args.max_epoch}')
        train_start_time = time()
        
        train_loss = 0.0
        train_iou = []
        train_conf = []
        train_validframe = []
        
        for batch, data in enumerate(tqdm(train_loader, leave=True)):
            # [batch, 3, h, w], [batch, 1, h, w]
            images, masks = data['images'].to(device), data['masks'].to(device)
            # print(images.dtype, masks.dtype)
            # print(images.shape, masks.shape)
            preds = model(images) # [batch, 1, h, w]
            # print(preds.shape)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Evaluate
            train_loss += loss.item()
            
            # calculate iou, conf, N_valid
            cal_batch_metric(preds, masks, train_iou, train_conf, train_validframe)
            # test append in fucntion work
            # print(len(train_iou))

        # score metrics
        train_wiou = np.mean(train_iou)
        train_atnr = np.mean(true_negative_curve(np.array(train_conf), np.array(train_validframe)))
        train_score = 0.7 * train_wiou + 0.3 * train_atnr
        
        train_loss_list.append(train_loss)
        train_score_list.append(train_score)
        train_time = time() - train_start_time
        # validation
        model.eval()
        valid_loss = 0.0
        valid_iou = []
        valid_conf = []
        valid_validframe = []

        with torch.no_grad():
            valid_start_time = time()
            for data in tqdm(valid_loader):
                images, masks = data['images'].to(device), data['masks'].to(device)
                
                preds = model(images)
                loss = criterion(preds, masks)

                valid_loss += loss.item()

                # calculate iou, conf, N_valid
                cal_batch_metric(preds, masks, train_iou, train_conf, train_validframe)

        valid_wiou = np.mean(valid_iou)
        valid_atnr = np.mean(true_negative_curve(np.array(valid_conf), np.array(valid_validframe)))
        valid_score = 0.7 * valid_wiou + 0.3 * valid_atnr
        
        valid_loss_list.append(valid_loss)
        valid_score_list.append(valid_score)
        valid_time = time() - valid_start_time

        # print each epoch's result: loss, score
        print(f'[{epoch + 1}/{args.max_epoch}] \
              {train_time:.2f}/{valid_time:.2f} sec(s) Score: {train_score:.3f}/{valid_score:.3f} | Loss: {train_loss:.3f}/{valid_loss:.3f}')
        
        # update scheduler
        scheduler.step()

        # check & update best loss
        is_better = valid_loss <= best_loss
        best_loss = valid_loss
        # save model
        if is_better:
            os.makedirs(args.model_save, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.model_save, f'model_best_{epoch}.pth'))
        # write textlog
        os.makedirs(args.log_save, exist_ok=True)
        write_txtlog(os.path.join(args.log_save, 'log.txt'), epoch, train_score, valid_score, train_loss, valid_loss\
                     , train_iou, valid_iou, train_atnr, valid_atnr, is_better)


if __name__ == '__main__':
    '''Choose Device'''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    
    '''Get DataLoader'''
    train_loader, valid_loader = get_dataloader(args.datapath, batch_size=args.batch_size, split='train', valid_ratio=0.1)
    # for batch, data in enumerate(train_loader):
    #     print(data['masks'].shape)
        # fig = plt.figure()
        # plt.imshow(data['masks'][0].permute(1,2,0))
        # plt.show()
        # print(type(data['images']), type(data['masks']))
        # print(data['images'].shape, data['masks'].shape)
        # print(len(data['masks'].split()))

    '''Pretrained Model Selection'''
    # https://github.com/mateuszbuda/brain-segmentation-pytorch
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)
    # https://github.com/milesial/Pytorch-UNet
    # model = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5)
    # model.outc.conv = Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1)) # unet_carvana outpur is 4 channel
    
    '''Model Setting & Summary'''
    model = nn.DataParallel(model, device_ids=[0, 1])
    model.to(device)
    summary(model, input_size=(3, 240, 320))
    # print(model)
    
    '''Selecting optimizer ...'''
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay)
    criterion = DiceLoss()
    # nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epoch)
    
    '''Start Training'''
    train(model=model,
          train_loader=train_loader,
          valid_loader=valid_loader,
          criterion=criterion,
          optimizer=optimizer,
          scheduler=scheduler
          )


    '''
    # This is useless #
    model.eval()
    input_image = Image.open('./dataset/S1/01/0.jpg').convert('RGB')

    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(input_image)
    print(input_tensor.shape)
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_batch)
        print(output.shape)
    out_np = output.cpu().numpy()
    print(np.unique(out_np))
    out = ((out_np -np.min(out_np)) / (np.max(out_np) - np.min(out_np) )* 255).astype(np.uint8)[0, 0]
    print(out.shape)
    # cv2.imshow("test", out)
    # cv2.waitKey(0)
    im = Image.fromarray(out.transpose(0, 1))
    im.save('./ans.png')
    '''