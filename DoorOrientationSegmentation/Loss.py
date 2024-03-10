import torch
import torchvision
from torch import nn

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        """
        paper: https://arxiv.org/pdf/1606.04797.pdf
        """
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, masks_pred, masks_true):
        masks_pred = torch.sigmoid(masks_pred)
        tp = masks_pred * masks_true
        fp = masks_pred * (1 - masks_true)
        fn = (1 - masks_pred) * masks_true
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()
        return -dc

# import numpy as np
# def dc(y_pred, y_true, smooth=1):
#     tp = y_pred * y_true
#     fp = y_pred * (1 - y_true)
#     fn = (1 - y_pred) * y_true
#     dc = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
#     return -dc

class CELoss(nn.Module):
    def __init__(self, device, pos_weight=None):
        super(CELoss, self).__init__()
        if pos_weight is None:
            self.cross_entropy = nn.BCEWithLogitsLoss()
        else:
            self.cross_entropy = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]).to(device))
    def forward(self, masks_pred, masks_true):
        return self.cross_entropy(masks_pred, masks_true)

if __name__ == '__main__':
    # be care to have the value befor sigmoid thresholding
    # loss_fn_unet = nn.BCEWithLogitsLoss()
    # loss_fn_unet = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([2.0]))

    loss_fn_unet = CELoss('cpu')
    # loss_fn_unet = SoftDiceLoss()
    masks_pred = torch.ones((4, 1, 256, 256), dtype=torch.float32) * 0.9 # sigmoid(0.9) = 0.71
    masks_true = torch.ones((4, 1, 256, 256), dtype=torch.float32)
    loss = loss_fn_unet(masks_pred, masks_true)
    print("masks_pred=0.71, masks_true=1: ", loss)

    masks_pred = torch.ones((4, 1, 256, 256), dtype=torch.float32) * -0.9 # sigmoid(-0.9) = 0.29
    masks_true = torch.zeros((4, 1, 256, 256), dtype=torch.float32)
    loss = loss_fn_unet(masks_pred, masks_true)
    print("masks_pred=0.29, masks_true=0: ", loss)

    masks_pred = torch.ones((4, 1, 256, 256), dtype=torch.float32) * -0.9
    masks_true = torch.ones((4, 1, 256, 256), dtype=torch.float32)
    loss = loss_fn_unet(masks_pred, masks_true)
    print("masks_pred=0.29, masks_true=1: ", loss)

    masks_pred = torch.ones((4, 1, 256, 256), dtype=torch.float32) * 0.9
    masks_true = torch.zeros((4, 1, 256, 256), dtype=torch.float32)
    loss = loss_fn_unet(masks_pred, masks_true)
    print("masks_pred=0.71, masks_true=0: ", loss)
