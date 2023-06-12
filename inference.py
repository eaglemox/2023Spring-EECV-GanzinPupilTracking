import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import *
from loss import *
from eval import *
from utils import *
from PIL import Image
from tqdm import tqdm
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)

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

def get_inference_dataloader(data_dir, batch_size):
    image_paths = get_sequence_path(data_dir)

    transform = transforms.Compose([
        transforms.Resize((240 ,320)),
        transforms.ToTensor(),
    ])
    dataset = InferenceGanzinDataset(image_paths, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    return dataloader

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

if __name__ == '__main__':
    '''Choose Device'''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    
    '''Fix Random Seed'''
    set_seed(2023)
    
    '''Hyperparameters'''
    batch_size = 64

    # model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=False)
    model = RTFNet(n_class=1)
    # model = UNet(n_channels=3, n_classes=1)
    '''Read Parameters (.pth)'''
    best_model = './test8/model_best_49.pth'
    # print(torch.load(best_model))
    model.load_state_dict(torch.load(best_model))
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[0, 1]) # enable two GPU parallel
    model.to(device)

    '''Inference S5-S8'''
    data_path = './dataset'
    output_path = './mask_9'
    # read to sequence level
    # for subject in ['S1', 'S2', 'S3', 'S4']:
    for subject in ['S5', 'S6', 'S7', 'S8']:
        print(f'Processing {subject}...')
        subject_path = os.path.join(data_path, subject)
        inference_loader = get_inference_dataloader(subject_path,batch_size)

        with torch.no_grad():
            for batch, data in enumerate(tqdm(inference_loader)):
                # open image
                images = data['images'].to(device)
                paths = data['paths']

                # inference
                preds = model(images)

                for batch in range(preds.shape[0]):
                    preds_tmp = preds.cpu().detach().numpy()
                    predict = preds_tmp[batch]
                    # print(predict.shape)
                    # predict = cv2.medianBlur(predict,5)
                    file_path = paths[batch]

                    # predict mask thresholding                    
                    predict_threshold = 0.99
                    predict = np.where(predict > predict_threshold, 1, 0)
                    if np.sum(predict) > 0:
                        conf = 1.0
                    else:
                        conf = 0.0
                    
                    # formatting RGB .png image
                    _, h, w = predict.shape
                    mask = np.zeros((h, w, 3))
                    mask[:, :, 0] = predict[0]
                    mask[:, :, 2] = predict[0]
                    mask = cv2.resize(mask, (640, 480), interpolation=cv2.INTER_CUBIC)
                    mask = np.where(mask > 0.99, 255, 0).astype(np.uint8)
                    # print(np.unique(mask))
                    # mask = (mask * 255).astype(np.uint8)
                    
                    # save mask.png
                    save_path = file_path.replace('.jpg', '.png').replace(data_path, output_path)
                    os.makedirs(os.path.split(save_path)[0], exist_ok=True)
                    cv2.imwrite(save_path, mask)
                    
                    # open (create), write conf.txt
                    conf_path, image_name = os.path.split(save_path) # "./output_path/Sx/xx", x.jpg
                    os.makedirs(conf_path, exist_ok=True)
                    f = open(f'{conf_path}/conf.txt', 'a')
                    f.write(f'{conf}\n')
                    f.close()
