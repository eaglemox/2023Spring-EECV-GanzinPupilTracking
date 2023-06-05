import cv2
import numpy as np
import math
from eval import *
def post(mask):
    '''
    mask 為 opencv h w channel
    if cv2.findContours抱錯 改成
    contours,_ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    '''
    # w1 = 0.5
    # w2 = 0.5
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    opening = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel, iterations=2)
    edged = cv2.Canny(opening, 30, 200)

    h, w = gray.shape
    # _, contours,_ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours,_ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img = np.zeros(mask.shape, dtype = np.uint8)
    valuelist  = []
    for contour in contours:
        if(len(contour) > 10):
            ellipse = cv2.fitEllipse(contour)
            block = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.ellipse(block, ellipse, (255,255,255), -1)

            value = np.sum(np.where(np.logical_and(block == 255, mask == 255), 1, 0))/3
            S2 =math.pi*ellipse[1][0]*ellipse[1][1]/4
            if S2 > 700:
                valuelist.append(value/S2)
            else:
                valuelist.append(-1)
        else:
            valuelist.append(-1)
    
    if not valuelist:
        return img, 0.0
    else:
        num = np.argmax(valuelist)
        if(valuelist[num] == -1):
            return img, 0.0
        ellipse = cv2.fitEllipse(contours[num])
        cv2.ellipse(img, ellipse, (255, 0, 255), -1)
        return img, 1.0
    
if __name__ == '__main__':
    data_path = "./S14_mask"
    iou_meter = AverageMeter()
    iou_meter_sequence = AverageMeter()
    label_validity = []
    output_conf = []
    for subject in tqdm(sorted(os.listdir(data_path))):
        # if subject in ['S1']:
        if subject in ['S1', 'S2', 'S3', 'S4']:
            for sequence in sorted(os.listdir(f'{data_path}/{subject}')):
                seq_image = []
                seq_mask = []
                f = open(f'{data_path}/{subject}/{sequence}/conf.txt', 'r')
                # conf = list(f.read)
                # idx = 0
                for name in sorted(os.listdir(f'{data_path}/{subject}/{sequence}')):
                    ext = os.path.splitext(name)[-1]
                    fullpath = f'{data_path}/{subject}/{sequence}/{name}'
                    os.makedirs(f'./post/{subject}/{sequence}', exist_ok=True)
                    if ext == '.png':
                        
                        output = cv2.imread(fullpath)
                        output, conf = post(output)
                        label = cv2.imread(fullpath.replace("S14_mask", "dataset"))
                        cv2.imwrite(fullpath.replace("S14_mask", "post"), output)
                        if np.sum(label.flatten()) > 0:
                            iou = mask_iou(output, label)
                            iou_meter.update(conf * iou)
                            iou_meter_sequence.update(conf * iou)
                            label_validity.append(1.0)
                            # idx += 1
                        else:
                            label_validity.append(0.0)
                        output_conf.append(conf)
    tn_rates = true_negative_curve(np.array(output_conf), np.array(label_validity))
    wiou = iou_meter.avg()
    atnr = np.mean(tn_rates)
    score = 0.7 * wiou + 0.3 * atnr
    print(f'\n\nOverall weighted IoU: {wiou:.4f}')
    print(f'Average true negative rate: {atnr:.4f}')
    print(f'Benchmark score: {score:.4f}')
