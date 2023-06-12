import os
import torch
import cv2
import numpy as np

from loss import *
from eval import *
from utils import *
from post import *
from PIL import Image
from tqdm import tqdm
from dataset import get_inference_dataloader
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

if __name__ == '__main__':
    '''Choose Device'''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    
    '''Fix Random Seed'''
    set_seed(2023)
    
    '''Hyperparameters'''
    batch_size = 64

    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=False)
    
    '''Read Parameters (.pth)'''
    best_model = './test8/model_best_49.pth'
    # print(torch.load(best_model))
    model.load_state_dict(torch.load(best_model))
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[0, 1]) # enable two GPU parallel
    model.to(device)

    '''Inference S5-S8'''
    data_path = './dataset'
    output_path = './mask_t2'
    # read to sequence level
    for subject in ['S1', 'S2', 'S3', 'S4']:
    # for subject in ['S5', 'S6', 'S7', 'S8']:
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

                    # mask, conf = post(mask)

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
