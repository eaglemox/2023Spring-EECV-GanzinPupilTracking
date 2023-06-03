import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-5):
        # inputs: float [0, 1], targets: float [0, 1]
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class DiceCeLoss(nn.Module):
    def __init__(self, dice_weight):
        super(DiceCeLoss, self).__init__()
        self.dice_weight = dice_weight
        self.ce_loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, pred, target):
        loss_dice = self.dice_loss(pred, target)
        loss_ce = self.ce_loss(pred, target)
        loss = (1 - self.dice_weight) * loss_ce + self.dice_weight * loss_dice

        return loss