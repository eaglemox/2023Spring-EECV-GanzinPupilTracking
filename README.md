# Computer Vison Final Project - Ganzin Pupil Tracking
## File Structure
If using default folder naming
``` 
Root
 ├─ dataset/: Ganzin pupil dataset
 │    ├─ S1/
 │    ├─ ...
 │    └─ S8/
 ├─ test/: model.pth & training log
 ├─ mask/: model's raw prediction
 ├─ solution/: result after post-processing
 ├─ *.py: codes
 └─ *.sh: bash files to run our codes (see below)
```
## Requirements
**Python version: 3.10.11**
- albumentations
- gdown
- matplotlib
- opencv-python
- opencv-contrib-python
- torch
- torchaudio
- torchsummary
- torchvision

Please refers to `requirements.txt` for more details. \
To install the required packages, run the command below.
```
pip install -r requirements.txt
```
## Download Dataset
Run `get_dataset.sh` to download dataset from google drive, then unzip it.
``` bash
bash get_dataset.sh
```
Or manually download from [here](https://drive.google.com/file/d/1XniSRxen6Ne7TMzFKzdax6xiWJKw_7SD/view?usp=drive_link) then unzip it, our default dataset path should be `./dataset`.
## Training
Run `train.sh`, default model's .pth file and training log will be saved to `./test` folder.
``` bash
bash train.sh
```
## Inference(Testing)
Manually edit model_path argument in `inference.sh` to choose the best model then run `inference.sh`.
``` bash
bash inference.sh
```
## Post-Processing
run `post_process.sh`, the final result will be saved in `./solution` folder. \
Contain pupil mask & conf.txt
``` bash
bash post_process.sh
```