import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import albumentations as A
from torch import optim
from model import *
from loss import *
# from losses import *
from eval import *
from utils import *
# from bl_utils import *
from config import args
from PIL import Image
from time import time
from tqdm import tqdm
from torchsummary import summary
from torch.nn import Conv2d
from torch.nn.functional import softmax
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR
from torchvision import transforms
from albumentations.pytorch.transforms import ToTensorV2

plt.switch_backend('agg')

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
                for name in sorted(os.listdir(f'{data_path}/{subject}/{sequence}')):
                    ext = os.path.splitext(name)[-1]
                    fullpath = f'{data_path}/{subject}/{sequence}/{name}'
                    if ext == '.jpg':
                        image_list.append(fullpath)
                    elif ext == '.png':
                        mask_list.append(fullpath)

    print(f'Number of image:{len(image_list)}, mask:{len(mask_list)}')
    return np.array(image_list), np.array(mask_list)

def get_dataloader(data_dir, batch_size, split='test', valid_ratio=0.1):
    image_path, mask_path = get_train_path(data_dir)
    if split == 'train':
        transform = A.Compose([
            A.Resize(480,640),
            A.Rotate(limit=20, p=0.3, border_mode=cv2.BORDER_CONSTANT),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.RandomGamma(gamma_limit=(20, 60), p=1.0), # 40 mean gamma = 40 / 100 = 0.4
            A.Normalize(mean=0.5, std=0.5),
            # A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.3, border_mode=cv2.BORDER_CONSTANT),
            # A.Equalize(p=0.3),
            # A.HueSaturationValue(p=0.4),
            # # A.Sharpen(p=0.3),
            # A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.3),
            # ToTensorV2(),

            ])
                                
        img_transform = transforms.Compose([
            transforms.Resize((240,320)),
            transforms.RandomEqualize(p=1.0),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.3, contrast= 0.5),
            transforms.ToTensor(),
            # TODO: add other augmentations
        ])
        mask_transform = transforms.Compose([
            transforms.Resize((240,320)),
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
        valid_loader = DataLoader(valid_set, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
        return train_loader, valid_loader
    else:
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
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
        trans = transforms.Compose([transforms.ToTensor()])
        transformed = self.transform(image=np.array(image), mask=np.array(mask))
        image, mask = trans(transformed['image']), trans(transformed['mask'])
        
        return {
            'images': image,
            'masks': mask
        }

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
            

def write_txtlog(log_path, current_epoch, train_score, valid_score, train_loss, valid_loss, train_wiou, valid_wiou, train_atnr, valid_atnr, is_better):
    with open(log_path, 'a') as f:
        f.write(f'[{current_epoch+1}/{args.max_epoch}] Score:{train_score:.5f}/{valid_score:.5f} | Loss:{train_loss:.5f}/{valid_loss:.5f} | ') # change line
        f.write(f'IOU:{train_wiou:.5f}/{valid_wiou:.5f} | ATNR:{train_atnr:.5f}/{valid_atnr:.5f}')
        if is_better:
            f.write('--> Best Updated')
        f.write('\n')

# def write_txtlog(log_path, current_epoch, train_score,train_loss, train_wiou, train_atnr, is_better):
#     with open(log_path, 'a') as f:
#         f.write(f'[{current_epoch+1}/{args.max_epoch}] Score:{train_score:.5f}  | Loss:{train_loss:.5f}/ | ') # change line
#         f.write(f'IOU:{train_wiou:.5f} | ATNR:{train_atnr:.5f}')
#         if is_better:
#             f.write('--> Best Updated')
#         f.write('\n')

def plot_learning_curve(results):
    for key, value in results.items():
        plt.plot(range(len(value)), value, label=f'{key}')
        plt.xlabel('Epoch')
        plt.ylabel(f'{key}')
        plt.title(f'Learning curve of {key}')
        plt.legend()

        plt.savefig(os.path.join(args.log_save, f'{key}.png'))
        plt.close()

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
            images, masks = data['images'].to(device), data['masks'].to(device)
            # print(images.dtype, masks.dtype)
            # print(images.shape, masks.shape)

            # print(f'label:{torch.unique(masks[0])}')
            preds = model(images) # [batch, 1, h, w]

            # check predict value distribution
            # for img in range(preds.shape[0]):
            #     print(f'pred:{torch.unique(preds[0])}')
            #     print(f'{img}')
            #     print(f'small count:{torch.count_nonzero(preds[img] < .99)}, large count:{torch.count_nonzero(preds[img] > .99)}')
            #     torch.count_nonzero(preds[img] > 0) == 0
            #     if True:
            #         plt.subplot(1, 3, 1)
            #         plt.imshow(images[img, 0].cpu().detach().numpy())
            #         plt.subplot(1, 3, 2)
            #         plt.imshow(masks[img, 0].cpu().detach().numpy())
            #         plt.subplot(1, 3, 3)
            #         plt.imshow(preds[img, 0].cpu().detach().numpy())
                    
            #         plt.show()
            
            # print(preds.shape)
            loss = criterion(preds, masks)
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
                images, masks = data['images'].to(device), data['masks'].to(device)
                
                preds = model(images)

                loss = criterion(preds, masks)

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
    # model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)
    model = RTFNet(n_class=1)
    # model = UNet(n_channels=3, n_classes=2)
    # model.conv = Conv2d(32, 2, kernel_size=1)
    # https://github.com/milesial/Pytorch-UNet
    # model = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5)
    # model.outc.conv = Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1)) # unet_carvana outpur is 4 channel
    
    '''Model Setting & Summary'''
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[0, 1]) # enable two GPU parallel
    model.to(device)

    # Show summary
    # summary(model, input_size=(3, 240, 320))
    # print(model)
    
    '''Selecting optimizer ...'''
    # optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay)
    # criterion = DiceCeLoss(dice_weight=0.5)
    # criterion = DiceBCELoss()
    criterion = nn.MSELoss()
    # GeneralizedDice(idc=[0])
    # BinaryDiceLoss()
    # nn.BCEWithLogitsLoss()
    # nn.CrossEntropyLoss()
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
