import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

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

def get_file_path(path='./dataset'):
    data_path = path
    image_list = []
    print('Acquiring inference image & mask path...')
    for name in sorted(os.listdir(data_path)):
        ext = os.path.splitext(name)[-1]
        fullpath = f'{data_path}/{name}'
        if ext == '.jpg':
            image_list.append(fullpath)
    print(f'Number of image:{len(image_list)}')

    return np.array(image_list)

def get_inference_dataloader(data_dir, batch_size):
    image_paths = get_file_path(data_dir)

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
    batch_size = 128

    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=False)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[0, 1]) # enable two GPU parallel
    model.to(device)
    
    '''Read Parameters (.pth)'''
    best_model = './test1/model_best_39.pth'
    # print(torch.load(best_model))
    model.load_state_dict(torch.load(best_model))
    

    '''Inference S5-S8'''
    data_path = './dataset'
    output_path = './solution'
    # read to sequence level
    for subject in ['S5', 'S6', 'S7', 'S8']:
        for sequence in sorted(os.listdir(f'{data_path}/{subject}')):
            sequence_path = os.path.join(data_path, subject, sequence)
            inference_loader = get_inference_dataloader(sequence_path,batch_size)

            # open (create) conf.txt
            cur_output_path = os.path.join(output_path, subject, sequence)
            os.makedirs(cur_output_path, exist_ok=True)
            f = open(f'{cur_output_path}/conf.txt', 'w')

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

                        file_path = paths[batch]
                        
                        predict_threshold = 0.99
                        predict = np.where(predict > predict_threshold, 1, 0)
                        if np.sum(predict) > 0:
                            conf = 1.0
                        else:
                            conf = 0.0
                        
                        # formatting RGBA image
                        _, h, w = predict.shape
                        mask = np.zeros((h, w, 3))
                        mask[:, :, 0] = predict[0]
                        mask[:, :, 2] = predict[0]
                        mask = (mask * 255).astype(np.uint8)
                        # no need v
                        # mask = cv2.cvtColor(mask, cv2.COLOR_RGB2RGBA)
                        mask = cv2.resize(mask, (640, 480), interpolation=cv2.INTER_CUBIC)
                        
                        save_path = file_path.replace('.jpg', '.png').replace('./dataset', './solution')
                        # save mask.png & conf.txt
                        f.write(f'{conf}\n')
                        os.makedirs(os.path.split(save_path)[0], exist_ok=True)
                        cv2.imwrite(save_path, mask)




        
    
