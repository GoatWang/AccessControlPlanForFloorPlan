import cv2
import torch
import random
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ToTensorCustom(A.BasicTransform):
    """Convert image and mask to `torch.Tensor`
    * Image numpy: [H, W, C] -> Image tensor: [C, H, W]
    * Mask numpy: [H, W, 1] -> Mask tensor: [1, H, W]
    """
    def __init__(self, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)

    @property
    def targets(self):
        return {"image": self.apply, "mask": self.apply_to_mask}

    def apply(self, img, **params):
        """Image from numpy [H, W, C] to tensor [C, H, W]"""
        if img.ndim == 2:
            img = np.expand_dims(img, axis=0)
        elif mask.ndim == 3:
            # [H, W, C] to tensor [C, H, W] in case mask has C > 1
            img = img.transpose(2, 0, 1)
        else:
            raise ValueError('img should have shape [H, W] without, '
                             'channel however provided mask shape was: '
                             '{}'.format(mask.shape))
        return torch.from_numpy(img.astype(np.float32))

    def apply_to_mask(self, mask, **params):
        """Mask from numpy [H, W] to tensor [1, H, W]"""
        # Adding channel to first dim if mask has no channel
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=0)
        # Transposing channel to channel first if mask has channel
        elif mask.ndim == 3:
            # [H, W, C] to tensor [C, H, W] in case mask has C > 1
            mask = mask.transpose(2, 0, 1)
        else:
            raise ValueError('Mask should have shape [H, W] without, '
                             'channel however provided mask shape was: '
                             '{}'.format(mask.shape))
        return torch.from_numpy(mask.astype(np.float32))


def get_transform(train, img_size, p=0.5):
    transforms = []
    if train:
        transforms.append(A.HorizontalFlip())
        # transforms.append(A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_upper=3, p=p))
        transforms.append(A.ShiftScaleRotate(scale_limit=(-0.4, 0.4), 
                                            rotate_limit=(-30, 30), 
                                            shift_limit_x=(-0.1, 0.1), 
                                            shift_limit_y=(-0.1, 0.1),
                                            # border_mode=1,  # cv2.BORDER_REPLICATE
                                            border_mode=0, #cv2.BORDER_CONSTANT,
                                            value=1, 
                                            p=p,
                                            ))
        transforms.append(A.Blur(blur_limit=10, p=p))
        transforms.append(A.Sharpen(alpha=(0.1, 0.3), lightness=(0.5, 1.0), p=p))
        transforms.append(A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.1, hue=0.05, p=p))
        transforms.append(A.GaussNoise(var_limit=0.01, p=p))
        
    transforms.append(A.Resize(*img_size))
    transforms.append(ToTensorCustom())
    return A.Compose(transforms)

if __name__ == '__main__':
    def turn_off_ticks(ax):
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        ax.set_xticks([])
        ax.set_yticks([])        
    
    import os
    import cv2
    import glob
    import torch
    import numpy as np
    from Dataset import read_img_mask
    from matplotlib import pyplot as plt
    if not os.path.exists('temp'):
        os.mkdir('temp')
    img_fp = sorted(glob.glob(os.path.join('DataDoorDirectionLabeled', 'train', 'images' , "*.jpg")))[0]
    txt_fp = sorted(glob.glob(os.path.join('DataDoorDirectionLabeled', 'train', 'labels' , "*.txt")))[0]
    X, mask = read_img_mask(img_fp, txt_fp)
    
    img_size= (128, 128)
    transform = get_transform(True, img_size)
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    axes = axes.flatten()
    for i in range(32):
        res = transform(image=X, mask=mask)
        X_draw = (res["image"].cpu().detach().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        X_draw = cv2.cvtColor(X_draw, cv2.COLOR_GRAY2BGR)
        X_mask = (res["mask"].cpu().detach().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        
        if np.sum(X_mask) > 0:
            contours, hierarchy = cv2.findContours(image=X_mask[:, :, 0], mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(image=X_draw, contours=contours, contourIdx=-1, color=(255, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
        axes[i].imshow(X_draw, vmin=0, vmax=1)
        turn_off_ticks(axes[i])
        
    plt.tight_layout()
    plt.savefig(os.path.join('temp', 'TransformFunctionsPreview_no_mask.png'))
    print("write file to", os.path.abspath(os.path.join('temp', 'TransformFunctionsPreview_no_mask.png')))

