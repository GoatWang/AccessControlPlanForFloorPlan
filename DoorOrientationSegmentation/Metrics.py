import torch
import torchvision
import torch.nn as nn

def cal_iou(masks_pred, masks_true, threshold=0.5, eps=1e-7):
    masks_pred = (masks_pred > threshold).type(masks_pred.dtype)
    tp = torch.sum(masks_pred * masks_true, axis=[1, 2, 3])
    fp = torch.sum(masks_pred * (1 - masks_true), axis=[1, 2, 3])
    fn = torch.sum((1 - masks_pred) * masks_true, axis=[1, 2, 3])
    ious = (tp + eps) / (tp + fp + fn + eps)
    return ious

class IoU(nn.Module):
    def __init__(self, threshold=0.5, channels=None, eps=1e-7):
        super(IoU, self).__init__()
        self.eps = eps
        self.threshold = threshold
        self.channels = channels
        channels_str = "_" + "_".join([str(c) for c in channels]) if channels is not None else 'all'
        self.__name__ = f'IoU_{int(self.threshold*100):d}' + channels_str

    def forward(self, masks_pred, masks_true):
        if self.channels is not None:
            masks_pred = masks_pred[:, self.channels, :, :]
            masks_true = masks_true[:, self.channels, :, :]
        masks_pred = torch.sigmoid(masks_pred)
        ious = cal_iou(masks_pred, masks_true, self.threshold, eps=self.eps)
        iou = ious.mean()
        return iou

class SoftIoU(nn.Module):
    def __init__(self, channels=None, eps=1e-7):
        super(SoftIoU, self).__init__()
        self.eps = eps
        self.channels = channels
        channels_str = "_" + "_".join([str(c) for c in channels]) if channels is not None else 'all'
        self.__name__ = f'SoftIoU' + channels_str


    def forward(self, masks_pred, masks_true):
        if self.channels is not None:
            masks_pred = masks_pred[:, self.channels, :, :]
            masks_true = masks_true[:, self.channels, :, :]
        masks_pred = torch.sigmoid(masks_pred)
        tp = masks_pred * masks_true
        fp = masks_pred * (1 - masks_true)
        fn = (1 - masks_pred) * masks_true
        soft_iou = (tp + self.eps) / (tp + fp + fn + self.eps)
        soft_iou = soft_iou.mean()
        return soft_iou

class AccuracyAtIoU(nn.Module):
    def __init__(self, threshold=0.5, iou_threshold=0.5, channels=None, eps=1e-7):
        super(AccuracyAtIoU, self).__init__()
        self.eps = eps
        self.channels = channels
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        channels_str = "_" + "_".join([str(c) for c in channels]) if channels is not None else 'all'
        self.__name__ = f'AccuracyAtIoU_{int(self.threshold*100):d}_{int(self.iou_threshold*100):d}' + channels_str

    def forward(self, masks_pred, masks_true):
        if self.channels is not None:
            masks_pred = masks_pred[:, self.channels, :, :]
            masks_true = masks_true[:, self.channels, :, :]
        masks_pred = torch.sigmoid(masks_pred)
        masks_true = masks_true.type(torch.int32)
        masks_pred = (masks_pred > self.threshold).type(torch.int32)
        tp = torch.sum(masks_pred * masks_true, axis=[1, 2, 3])
        fp = torch.sum(masks_pred * (1 - masks_true), axis=[1, 2, 3])
        fn = torch.sum((1 - masks_pred) * masks_true, axis=[1, 2, 3])
        iou = (tp + self.eps) / (tp + fp + fn + self.eps)
        return torch.sum((iou > self.iou_threshold)) / (masks_true.shape[0])

class BoxIoU(nn.Module):
    def __init__(self):
        super(BoxIoU, self).__init__()
        self.__name__ = f'BoxIoU'

    def forward(self, masks_pred, masks_true, boxes_pred, boxes_true):
        # box_ious = torchvision.ops.box_iou(boxes_pred.reshape(-1, 4), boxes_true.reshape(-1, 4))[torch.eye(boxes_true.reshape(-1, 4).shape[0]).type(torch.bool)]
        box_ious = []
        for box_pred, box_true in zip(boxes_pred.reshape(-1, 4), boxes_true.reshape(-1, 4)):
            box_iou = torchvision.ops.box_iou(box_pred[None, :], box_true[None, :])
            box_ious.append(box_iou)
        return torch.mean(torch.stack(box_ious))



if __name__ == "__main__":
    masks_pred = torch.ones([4, 1, 256, 256], dtype=torch.float32) * 10
    masks_true = torch.ones([4, 1, 256, 256], dtype=torch.float32)
    masks_pred[:, :, :100, :100] = -10
    iou_score = IoU()(masks_pred, masks_true)
    iou_acuracy_score = AccuracyAtIoU()(masks_pred, masks_true)
    print("iou_score", iou_score)
    print("iou_acuracy_score", iou_acuracy_score)

    masks_pred = torch.ones((4, 1, 256, 256), dtype=torch.float32) * 0.9
    masks_true = torch.ones((4, 1, 256, 256), dtype=torch.float32)
    iou_score = IoU()(masks_pred, masks_true)
    iou_acuracy_score = AccuracyAtIoU()(masks_pred, masks_true)
    print("iou_score", iou_score)
    print("iou_acuracy_score", iou_acuracy_score)

    masks_pred = torch.ones((4, 1, 256, 256), dtype=torch.float32) * -0.9
    masks_true = torch.zeros((4, 1, 256, 256), dtype=torch.float32)
    iou_score = IoU()(masks_pred, masks_true)
    iou_acuracy_score = AccuracyAtIoU()(masks_pred, masks_true)
    print("iou_score", iou_score)
    print("iou_acuracy_score", iou_acuracy_score)

    # boxes_pred = torch.stack([torch.Tensor([0, 0, 0.9, 0.9]).type(torch.float32)[None, :] for i in range(4)]).reshape(-1, 4)
    # boxes_true = torch.stack([torch.Tensor([0, 0, 1, 1]).type(torch.float32)[None, :] for i in range(4)]).reshape(-1, 4)
    # print(boxes_true.shape)
    boxes_pred = torch.Tensor([0, 0, 0.7, 0.7, 0, 0, 0.8, 0.8, 0, 0, 0.9, 0.9]).type(torch.float32).reshape(1, 3, 4)
    boxes_true = torch.Tensor([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]).type(torch.float32).reshape(1, 3, 4)
    box_iou_score = BoxIoU()(masks_pred, masks_true, boxes_pred, boxes_true)
    print("box_iou_score", box_iou_score)

