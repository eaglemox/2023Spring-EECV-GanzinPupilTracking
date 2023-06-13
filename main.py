import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from loss import *
from eval import *
from model import *
from utils import *
from config import args
from time import time
from tqdm import tqdm
from torch.nn import Conv2d
from torchsummary import summary
from datasets import get_dataloader
from torch.nn.functional import softmax
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR

plt.switch_backend('agg')

def cal_batch_metric(predict_batch, label_batch, iou_list, conf_list, validframe_list):
    predict_batch = predict_batch.cpu().detach().numpy()
    label_batch = label_batch.cpu().detach().numpy()

    for batch in range(predict_batch.shape[0]):
        # shape (c, h, w)
        predict = predict_batch[batch]
        label = label_batch[batch]
        # threshold predict, 0/~1
        predict_threshold = 0.99
        predict = np.where(predict > predict_threshold, 1, 0)
        if np.sum(predict) > 0:
            conf = 1.0
            # mask_iou input shape (h, w, c)
            iou = mask_iou(predict.transpose(1, 2, 0), label.transpose(1, 2, 0))
            iou_list.append(conf * iou)
        else:  # empty ground truth label
            conf = 0.0

        conf_list.append(conf)

        if np.sum(label) > 0:
            validframe_list.append(1.0)
        else:  # empty ground truth label
            validframe_list.append(0.0)

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
        
        for batch, data in enumerate(tqdm(train_loader)):
            # [batch, 3, h, w], [batch, 1, h, w]
            images, masks, dists = data['images'].to(device), data['masks'].to(device), data['dists'].to(device)
            # print(images.dtype, masks.dtype)
            # print(images.shape, masks.shape)

            # print(f'label:{torch.unique(masks[0])}')
            preds = model(images) # [batch, 1, h, w]
            # preds = softmax(preds, dim=1)
            
            # print(preds.shape)
            loss = criterion(preds, masks, dists)
            # print(f'loss:{loss:.3f}')
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
        train_wiou = np.sum(train_iou) / len(train_validframe)
        train_atnr = np.mean(true_negative_curve(np.array(train_conf), np.array(train_validframe)))
        train_score = 0.7 * train_wiou + 0.3 * train_atnr
        
        train_loss = train_loss / len(train_loader)
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
                images, masks, dists = data['images'].to(device), data['masks'].to(device), data['dists'].to(device)
                
                preds = model(images)
                # preds = softmax(preds, dim=1)

                loss = criterion(preds, masks, dists)

                valid_loss += loss.item()

                # calculate iou, conf, N_valid
                cal_batch_metric(preds, masks, valid_iou, valid_conf, valid_validframe)

        valid_wiou = np.sum(valid_iou) / len(valid_validframe)
        valid_atnr = np.mean(true_negative_curve(np.array(valid_conf), np.array(valid_validframe)))
        valid_score = 0.7 * valid_wiou + 0.3 * valid_atnr
        
        valid_loss = valid_loss / len(valid_loader)
        valid_loss_list.append(valid_loss)
        valid_score_list.append(valid_score)
        valid_time = time() - valid_start_time

        # print each epoch's result: loss, score
        print(f'[{epoch + 1}/{args.max_epoch}] {train_time:.2f}/{valid_time:.2f} sec(s) Score: {train_score:.3f}/{valid_score:.3f} | Loss: {train_loss:.3f}/{valid_loss:.3f}')
        # print(f'[{epoch + 1}/{args.max_epoch}] {train_time:.2f} sec(s) Score: {train_score:.3f} | Loss: {train_loss:.3f}')
        
        
        # update scheduler
        scheduler.step()

        # check & update best loss
        is_better = train_loss <= best_loss
        best_loss = train_loss

        # save model
        if is_better:
            os.makedirs(args.model_save, exist_ok=True)
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), os.path.join(args.model_save, f'model_best_{epoch}.pth'))
            else:
                torch.save(model.state_dict(), os.path.join(args.model_save, f'model_best_{epoch}.pth'))

        # write textlog
        os.makedirs(args.log_save, exist_ok=True)
        write_txtlog(os.path.join(args.log_save, 'log.txt'), epoch, train_score, valid_score, train_loss, valid_loss\
                     , train_wiou, valid_wiou, train_atnr, valid_atnr, is_better)
        # write_txtlog(os.path.join(args.log_save, 'log.txt'), epoch, train_score, train_loss,\
        #              train_wiou, train_atnr, is_better)       
        # plot learning curve (every epoch)
        result_lists = {
                'train_score': train_score_list,
                'train_loss': train_loss_list,
                'valid_score': valid_score_list,
                'valid_loss': valid_loss_list
            }
        plot_learning_curve(result_lists)


if __name__ == '__main__':
    '''Choose Device'''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    
    '''Fix Random Seed'''
    set_seed(2023)

    '''Get DataLoader'''
    train_loader, valid_loader = get_dataloader(args.data_path, batch_size=args.batch_size, split='train', valid_ratio=0.1)

    '''Pretrained Model Selection'''
    # https://github.com/mateuszbuda/brain-segmentation-pytorch
    # model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)
    # model.conv = Conv2d(32, 2, kernel_size=1)
    model = RTFNet(n_class=1)
    # model = UNet(n_channels=3, n_classes=2)
    # https://github.com/milesial/Pytorch-UNet
    # model = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5)
    # model.outc.conv = Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1)) # unet_carvana outpur is 4 channel
    
    '''Model Setting'''
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[0, 1]) # enable two GPU parallel
    model.to(device)

    '''Show summary'''
    # summary(model, input_size=(3, 240, 320))
    # print(model)
    
    '''Selecting optimizer ...'''
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay)
    # criterion = DiceBCELoss()
    criterion = nn.MSELoss()
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epoch)
    
    '''Start Training'''
    train(model=model,
          train_loader=train_loader,
          valid_loader=valid_loader,
          criterion=criterion,
          optimizer=optimizer,
          scheduler=scheduler,
          )
