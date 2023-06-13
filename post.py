import os
import cv2
import math
import argparse
import numpy as np

from tqdm import tqdm

def post(img, mask, lastconf, x, y, limit=15, size=3, lowrate=0.1):
    '''
    input
    img: orignal image: (h, w, 3) np.uint8
    mask: model predict output mask: (h, w, 3) np.uint8
    lastconf: confidence number: int
    x: the center of ellispe in x direction in last frame: int, -1 means last frame have no mask
    y: the center of ellispe in y direction in last frame: int, -1 means last frame have no mask
    '''

    # calculate the beta value for image enhancement
    imgpro = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    value = -0.07*np.average(imgpro) + 13.5

    # image enhancement
    imgpro = cv2.convertScaleAbs(imgpro, alpha=3, beta=value)
    # image blur hope to remove the eyelash
    imgpro = cv2.medianBlur(imgpro, 5)
    # local historgram to enhance the different between iris and pupil
    clahe = cv2.createCLAHE(clipLimit=15, tileGridSize=(3,3))
    imgpro = clahe.apply(imgpro)
    # blur the image
    imgpro = cv2.GaussianBlur(imgpro, (7, 7), 0)

    # Canny (Otsu's threshold)
    ret, imgth = cv2.threshold(imgpro, thresh=0, maxval=255, type=(cv2.THRESH_BINARY + cv2.THRESH_OTSU))
    edged = cv2.Canny(imgpro, threshold1=(ret * 0.1), threshold2=ret)

    # turn the mask into one channel 
    mask1 = mask[:,:,0]

    # the postprocess for the mask that output from the model, 
    # remove the noise and fill up the hole by morphology
    kernel = np.ones((31, 31), np.uint8)
    kernel1 = np.ones((3,3), np.uint8)
    mask1 = cv2.erode(mask1, kernel1, iterations = 1)
    mask1 = cv2.dilate(mask1, kernel, iterations = 1)
    mask1 = cv2.erode(mask1, kernel, iterations = 1)

    # use the mask from model to remove the other edge which is not around to eye
    imgpro = np.where(mask1==255, edged, 0)

    # contour finding
    contours, _ = cv2.findContours(imgpro, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # for windows
    # _, contours, _ = cv2.findContours(imgpro, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    h,w,_ = img.shape
    valuelist = []
    ell = []

    for cnt in contours:
        #remove the noise
        if len(cnt) > 100:
            # fit the contour with ellipse
            ellipse = cv2.fitEllipse(cnt)
            block = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.ellipse(block, ellipse, (255,255,255), -1)
            block = block[:,:,0]

            # caluclate the black area inside the ellipse
            newmask2 = np.where(np.logical_and(imgth == 0, block == 255), 255, 0).astype(np.uint8)
            value = np.sum(np.where(np.logical_and(block == 255, newmask2 == 255), 1, 0))
            S2 =math.pi*ellipse[1][0]*ellipse[1][1]/4  

            # filter 1. size of the ellipse
            if S2 < 700 or S2 > 12000:
                pass 
            # the shape of the ellipse
            elif lastconf == 1 and ellipse[1][0]/ellipse[1][1] < 0.4:
                pass
            # the case that is half open the eye
            elif lastconf == 0 and ellipse[1][0]/ellipse[1][1] < 0.6:
                pass
            # the prediction that is to far away
            elif x != -1 and y != -1 and math.sqrt( (ellipse[0][0]-x)**2 + (ellipse[0][1]-y)**2 ) > 70:
                pass
            # save the ratio and the ellipse in a list
            else:
                valuelist.append(value/S2)
                ell.append(ellipse)
    # output the mask
    mask = np.zeros((h, w, 3), dtype = np.uint8)
    # case without any ellipse
    if not valuelist:
        # output the original gauss for the 1 0 1 predict case
        mask[:,:,0] = mask1
        mask[:,:,2] = mask1
        return mask, 0.0, -1, -1
    else:
        # get the maximum ratio
        num = np.argmax(valuelist)

        if valuelist[num] < 0.5:
            # with too small ratio is not consider as a pupil
            mask[:,:,0] = mask1
            mask[:,:,2] = mask1
            return mask, 0.0, -1, -1
        else:

            cv2.ellipse(mask, ell[num], (255, 0, 255), -1)
            return mask, 1.0, ell[num][0][0], ell[num][0][1]


def main(mask_path, data_path, save_path, testset):
    mask_path = mask_path
    data_path = data_path
    save_path = save_path
    os.makedirs(f'./{save_path}', exist_ok=True)

    subjects = []
    if testset:
        subjects = ['S5', 'S6', 'S7', 'S8']
    else:
        subjects = ['S1', 'S2', 'S3', 'S4']

    for subject in subjects:
        for sequence in tqdm(sorted(os.listdir(f'{data_path}/{subject}'))):
            os.makedirs(f'./{save_path}/{subject}/{sequence}', exist_ok=True)
            f = open(f'./{save_path}/{subject}/{sequence}/conf.txt', 'w')
            lastconf = 1
            last2conf = 1
            conflist = []
            x = -1
            y = -1 
            for name in sorted(os.listdir(f'{data_path}/{subject}/{sequence}'),\
                                key=lambda x: int(x.replace(".jpg", "").replace(".png", ""))):
                ext = os.path.splitext(name)[-1]
                fullpath = f'{data_path}/{subject}/{sequence}/{name}'
                os.makedirs(f'./{save_path}/{subject}/{sequence}', exist_ok=True)
                ext = os.path.splitext(name)[-1]
                if ext == '.jpg':
                    img = cv2.imread(fullpath)
                    mask = cv2.imread(fullpath.replace(data_path, mask_path).replace("jpg", "png"))

                    outmask, conf, x, y= post(img, mask, lastconf, x, y)
                    
                    cv2.imwrite(fullpath.replace(data_path, save_path).replace("jpg", "png"), outmask)
                    
                    # case 0 1 0
                    if last2conf==0 and conf == 0 and lastconf == 1:
                        del conflist[-1]
                        lastconf = 0
                        conflist.append(0.0)
                    # case 1 0 1
                    if lastconf==1 and conf == 1 and lastconf == 0:
                        del conflist[-1]
                        lastconf = 1
                        conflist.append(1.0)

                    last2conf = lastconf
                    lastconf = conf
                    conflist.append(conf)

            for i in range(len(conflist)):
                f.write(f'{conflist[i]}\n')
                        
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Post Processing')
    parser.add_argument('--mask_path', type=str, default='./mask_8', help='path to predicted mask folder')
    parser.add_argument('--data_path', type=str, default='./dataset', help='path to dataset folder')
    parser.add_argument('--save_path', type=str, default='./solution_test', help='folder to save post-processed results')
    parser.add_argument('--testset', type=bool, default=True, help='whether process S5-S8 data')
    args = parser.parse_args()
    
    main(args.mask_path, args.data_path, args.save_path, args.testset)