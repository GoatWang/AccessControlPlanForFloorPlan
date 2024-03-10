import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, pooling_size=(1, 1)):
        """
        convtype: {'down', 'up}
        """
        modules = []
        modules.append(nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        modules.append(nn.BatchNorm2d(out_channels))
        modules.append(nn.ReLU())
        if pooling_size != (1, 1):
            modules.append(nn.MaxPool2d(pooling_size))
        super(ConvBNReLU, self).__init__(*modules)
        

class DownScaling(nn.Sequential):
    def __init__(self, in_channels, out_channels, pooling_size=(2, 2), layers=2):
        """
        convtype: {'down', 'up}
        """
        modules = []
        modules.append(nn.MaxPool2d(pooling_size))
        modules.append(ConvBNReLU(in_channels, out_channels))
        for i in range(layers - 1):
            modules.append(ConvBNReLU(out_channels, out_channels))
        super(DownScaling, self).__init__(*modules)

class UpScaling(nn.Module):
    def __init__(self, in_channels, out_channels, pooling_size=(2, 2), layers=2):
        super(UpScaling, self).__init__()
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=pooling_size, stride=pooling_size)
        self.conv1 = ConvBNReLU(in_channels, out_channels)
        self.conv2 = ConvBNReLU(out_channels, out_channels)
        self.layers = layers
        
    def forward(self, x_up, x_down):
        """
        x_up: (52, 52, 64)
        x_down: (104, 104, 32)
        """
        x_up = self.up(x_up) # (64, 104, 104)
        x_up = torch.cat([x_up, x_down], dim=1) # (64+32, 104, 104)
        x_up = self.conv1(x_up) # (32, 104, 104)
        if self.layers == 2:
            x_up = self.conv2(x_up)
        return x_up
        

class Unet(nn.Module):
    ''' Models a simple Convolutional Neural Network'''
    def __init__(self, input_size, device):
        ''' initialize the network '''
        super(Unet, self).__init__()
        self.in_conv = ConvBNReLU(1, 32)
        self.down1 = DownScaling(32, 64, layers=4)
        self.down2 = DownScaling(64, 128, layers=4)
        self.down3 = DownScaling(128, 256, layers=4)
        self.down4 = DownScaling(256, 512, layers=4)
        self.up4 = UpScaling(512, 256, layers=4)
        self.up3 = UpScaling(256, 128, layers=4)
        self.up2 = UpScaling(128, 64, layers=4)
        self.up1 = UpScaling(64, 32, layers=4)
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        ''' the forward propagation algorithm '''
        x_in_conv = self.in_conv(x)
        x_down1 = self.down1(x_in_conv)
        x_down2 = self.down2(x_down1)
        x_down3 = self.down3(x_down2)
        x_down4 = self.down4(x_down3)
        x_up4 = self.up4(x_down4, x_down3)
        x_up3 = self.up3(x_up4, x_down2)
        x_up2 = self.up2(x_up3, x_down1)
        x_up1 = self.up1(x_up2, x_in_conv)
        x_out_conv = self.out_conv(x_up1)
        return x_out_conv


        
if __name__ == '__main__':
    import torch
    from pprint import pprint
    from torchsummary import summary
    input_size = (128, 128)
    device = 'cpu'
    model = Unet(input_size, device)
    model.to(device)
    summary(model, (1, *input_size), device=device)
    # images = torch.ones((1, 3, *input_size)).to(device)
    # masks = model(images)
    # pprint(masks.shape)




    # import os
    # import glob
    # from Transforms import get_transform
    # from matplotlib import pyplot as plt
    # from Dataset import AgingElectrodeSegmentationDataset
    # img_fps = sorted(glob.glob(os.path.join('data', 'training') + "/*_X.npy"))
    # mask_fps = sorted(glob.glob(os.path.join('data', 'training') + "/*_Y.npy"))
    # dataset_train = AgingElectrodeSegmentationDataset(img_fps, mask_fps, input_size, get_transform(train=True, img_size=input_size))

    # model_fp = 'models/unet/2021-12-29-05-39-12/38_0.010948_0.4893_0.9714_0.2141_0.6661_0.1922_0.7765.pt'
    # model.load_state_dict(torch.load(model_fp)['state_dict'])
    # model.to(device)
    # model.eval()

    # img_idx = 300
    # img, masks_true = dataset_train[img_idx]
    # img, masks_true = img.to(device)[None, :, :, :], masks_true.to(device)[None, :, :, :]
    # masks_pred = model(img)
    # x_out_conv = masks_pred[:, 2:8]

    # needle_short_box_mask_rectangle_batches = []
    # for batch_idx in range(x_out_conv.shape[0]):
    #     needle_short_box_mask = torch.sigmoid(x_out_conv[batch_idx, 4]) > 0.5
    #     if torch.sum(needle_short_box_mask) > 0:
    #         ys, xs = torch.where(needle_short_box_mask)
    #         xmin, ymin, xmax, ymax = xs.min(), ys.min(), xs.max(), ys.max()
    #     else:
    #         xmin, ymin, xmax, ymax = 0, 0, 128, 128
    #     needle_short_box_mask_rectangle = torch.zeros_like(needle_short_box_mask).type(torch.float32)
    #     needle_short_box_mask_rectangle[ymin:ymax, xmin:xmax] = 1
    #     needle_short_box_mask_rectangle_batches.append(needle_short_box_mask_rectangle[None, :, :])
    # needle_short_box_mask_rectangle_batches = torch.stack(needle_short_box_mask_rectangle_batches)

    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
    # ax1.imshow(img[0, 0, :, :], cmap='gray')
    # ax2.imshow(needle_short_box_mask_rectangle_batches[0, 0, :, :], cmap='gray')
    # ax3.imshow(img[0, 0, :, :] * needle_short_box_mask_rectangle_batches[0, 0, :, :], cmap='gray')
    # plt.savefig(os.path.join('temp', 'ModelMiddleLayerInspect.png'))
    # print("ModelMiddleLayerInspect write to", os.path.join('temp', 'ModelMiddleLayerInspect.png'))
















# class ResidualConvBNReLU(nn.Module):
#     def __init__(self, out_channels):
#         super(ResidualConvBNReLU, self).__init__()
#         self.relu = nn.ReLU(inplace=True)
#         self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         self.bn2 = nn.BatchNorm2d(out_channels)
 
#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = torch.add(out, residual)
#         out = self.relu(out)
#         return out


# class DownScaling(nn.Sequential):
#     def __init__(self, in_channels, out_channels, pooling_size=(2, 2)):
#         """
#         convtype: {'down', 'up}
#         """
#         super(DownScaling, self).__init__(
#             nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
#             ResidualConvBNReLU(out_channels),
#             nn.MaxPool2d(pooling_size),
#         )

# class UpScaling(nn.Module):
#     def __init__(self, in_channels, out_channels, pooling_size=(2, 2)):
#         super(UpScaling, self).__init__()
#         # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=pooling_size, stride=pooling_size)
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         self.residual_block = ResidualConvBNReLU(out_channels)

#     def forward(self, x_up, x_down):
#         """
#         x_up: (52, 52, 64)
#         x_down: (104, 104, 32)
#         """
#         x_up = self.up(x_up) # (64, 104, 104)
#         x_up = torch.cat([x_up, x_down], dim=1) # (64+32, 104, 104)
#         x_up = self.conv(x_up)
#         x_up = self.residual_block(x_up)
#         return x_up
        
# class Unet(nn.Module):
#     ''' Models a simple Convolutional Neural Network'''
#     def __init__(self, input_size, device):
#         ''' initialize the network '''
#         super(Unet, self).__init__()
#         row_idxs, col_idxs = torch.meshgrid([torch.arange(0, input_size[0]), torch.arange(0, input_size[1])])
#         row_idxs, col_idxs = row_idxs / row_idxs.max(), col_idxs / col_idxs.max()
#         self.x_pe = torch.cat([row_idxs[None, :, :], col_idxs[None, :, :]]).to(device)
#         self.in_conv = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         self.down1 = DownScaling(32, 64)
#         self.down2 = DownScaling(64, 128)
#         self.down3 = DownScaling(128, 256)
#         self.down4 = DownScaling(256, 512)
#         self.up4 = UpScaling(512, 256)
#         self.up3 = UpScaling(256, 128)
#         self.up2 = UpScaling(128, 64)
#         self.up1 = UpScaling(64, 32)
#         self.out_conv = nn.Conv2d(32, 6, kernel_size=1) # needle_short, needle_long, [jig_bound, ccd, jig, mattee, needle_short_box, needle_long_box]

#     def forward(self, x):
#         ''' the forward propagation algorithm '''
#         x_pe = torch.tile(self.x_pe[None, :, :, :], dims=[x.shape[0], 1, 1, 1])
#         x_in_conv = self.in_conv(torch.cat([x, x_pe], axis=1)) # (3, 256, 256) => (8, 128, 128)
#         x_down1 = self.down1(x_in_conv) # (8, 128, 128) => (16, 64, 64)
#         x_down2 = self.down2(x_down1) # (16, 64, 64) => (32, 32, 32)
#         x_down3 = self.down3(x_down2) # (32, 32, 32) => (64, 16, 16)
#         x_down4 = self.down4(x_down3) # (64, 16, 16) => (128, 8, 8)
#         x_up4 = self.up4(x_down4, x_down3) # (128, 8, 8) => (64, 16, 16)
#         x_up3 = self.up3(x_up4, x_down2) # (64, 16, 16) => (32, 32, 32)
#         x_up2 = self.up2(x_up3, x_down1) # (32, 32, 32) => (16, 64, 64)
#         x_up1 = self.up1(x_up2, x_in_conv) # (16, 64, 64) => (8, 128, 128)
#         x_out_conv = self.out_conv(x_up1) # (4, 128, 128) => (1, 128, 128)
#         return x_out_conv
