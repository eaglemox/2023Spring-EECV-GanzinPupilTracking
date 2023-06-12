import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import einsum
 
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
        self.dice_loss = DiceLoss()
        self.dice_weight = dice_weight
        self.ce_loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, pred, target):
        loss_dice = self.dice_loss(pred, target)
        loss_ce = self.ce_loss(pred, target)
        loss = (1 - self.dice_weight) * loss_ce + self.dice_weight * loss_dice

        return loss

class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()

    def forward(self, pred, target):
        # dist_map = mask2dist(target.cpu().detach().numpy())

        # print(pred.shape, target.shape)
        # dist_map = torch.tensor(dist_map, dtype=torch.float32)
        multipled = einsum("bkwh,bkwh->bwh", pred, target)
        
        loss = multipled.mean()

        return loss
        # pass
    
class DiceBoundaryCeLoss(nn.Module):
    def __init__(self, boundary_weight: float):
        super(DiceBoundaryCeLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.boundary_loss = BoundaryLoss()
        self.weight = boundary_weight

    def forward(self, pred, target, dist_map):
        loss_dice = self.dice_loss(pred, target)
        loss_ce = self.ce_loss(pred, target)
        loss_boundary = self.boundary_loss(pred, dist_map)

        loss = loss_dice + loss_ce + self.weight*loss_boundary

        return loss
