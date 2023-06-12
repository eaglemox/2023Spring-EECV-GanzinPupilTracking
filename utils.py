import os
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from config import args
from scipy.ndimage import distance_transform_edt as eucl_distance

plt.switch_backend('agg')

def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)

def mask2dist(mask):
    mask_tmp = mask.cpu().detach().numpy()
    K = mask_tmp.shape[0]
    # print(K)

    res = np.zeros_like(mask_tmp)
    for k in range(K):
        posmask = mask_tmp[k].astype(np.bool_)

        if posmask.any():
            negmask = ~posmask
            res[k] = eucl_distance(negmask) * negmask \
                - (eucl_distance(posmask) - 1) * posmask
    # print(f'func shape{res.shape}')
    return res

def write_txtlog(log_path, current_epoch, train_score, valid_score, train_loss, valid_loss, train_wiou, valid_wiou, train_atnr, valid_atnr, is_better):
    with open(log_path, 'a') as f:
        f.write(f'[{current_epoch+1}/{args.max_epoch}] Score:{train_score:.5f}/{valid_score:.5f} | Loss:{train_loss:.5f}/{valid_loss:.5f} | ') # change line
        f.write(f'IOU:{train_wiou:.5f}/{valid_wiou:.5f} | ATNR:{train_atnr:.5f}/{valid_atnr:.5f}')
        if is_better:
            f.write('--> Best Updated')
        f.write('\n')

def plot_learning_curve(results):
    for key, value in results.items():
        plt.plot(range(len(value)), value, label=f'{key}')
        plt.xlabel('Epoch')
        plt.ylabel(f'{key}')
        plt.title(f'Learning curve of {key}')
        plt.legend()

        plt.savefig(os.path.join(args.log_save, f'{key}.png'))
        plt.close()

def alpha_blend(input_image: np.ndarray, segmentation_mask: np.ndarray, alpha: float = 0.5):
    """Alpha Blending utility to overlay segmentation masks on input images
    Args:
        input_image: a np.ndarray with 1 or 3 channels
        segmentation_mask: a np.ndarray with 3 channels
        alpha: a float value
    """
    if len(input_image.shape) == 2:
        input_image = np.stack((input_image,) * 3, axis=-1)
    blended = input_image.astype(np.float32) * alpha + segmentation_mask.astype(np.float32) * (1 - alpha)
    blended = np.clip(blended, 0, 255)
    blended = blended.astype(np.uint8)
    return blended


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.sum = 0
        self.count = 0

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.sum += val
        self.count += 1

    def avg(self):
        return self.sum / self.count


if __name__ == '__main__':
    import os
    import cv2
    import matplotlib
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-S', '--subject', default='1')
    parser.add_argument('-Q', '--sequence', default='01')
    args = parser.parse_args()

    dataset_path = f'./dataset/S{args.subject}/{args.sequence}'
    mask_path = f'./mask_11/S{args.subject}/{args.sequence}'
    nr_image = len([name for name in os.listdir(dataset_path) if name.endswith('.jpg')])
    print(nr_image)
    image = cv2.imread(os.path.join(dataset_path, '0.jpg'))
    h = image.shape[0]
    w = image.shape[1]
    dpi = matplotlib.rcParams['figure.dpi']
    fig = plt.figure(figsize=(w / dpi, h / dpi))
    ax = fig.add_axes([0, 0, 1, 1])
    for batch in range(nr_image):
        image_name = os.path.join(dataset_path, f'{batch}.jpg')
        label_name = os.path.join(mask_path, f'{batch}.png')
        image = cv2.imread(image_name)
        label = cv2.imread(label_name)
        blended = alpha_blend(image, label, 0.5)
        ax.clear()
        ax.imshow(blended)
        ax.axis('off')
        plt.draw()
        plt.pause(0.01)
    plt.close()
