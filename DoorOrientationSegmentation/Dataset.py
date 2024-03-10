import os
import cv2
import glob
import torch
import numpy as np
from matplotlib import pyplot as plt

def read_box_as_mask(txt_fp, h_ori, w_ori):
    with open(txt_fp, "r") as f:
        box = []
        x, y, w, h = [float(elem) for elem in f.read().split("\n")[0].split()[1:]]
        x1, y1, x2, y2 = int((x-w/2)*w_ori), int((y-h/2)*h_ori), int((x+w/2)*w_ori), int((y+h/2)*h_ori)
        mask = np.zeros((h_ori, w_ori)).astype(np.uint8)

        color = (255, 255, 255)
        thickness = -1
        cv2.rectangle(mask, (x1, y1), (x2, y2), color, thickness)
    return mask
    
def read_img_mask(img_fp, txt_fp):
    X = cv2.cvtColor(cv2.imread(img_fp), cv2.COLOR_BGR2GRAY)
    h_ori, w_ori = X.shape[:2]
    mask = read_box_as_mask(txt_fp, h_ori, w_ori)
    return (X/255).astype(np.float32), (mask/255).astype(np.float32)
    
class DoorDataset(torch.utils.data.Dataset):
    def __init__(self, img_fps, txt_fps, img_size, transforms):
        self.transforms = transforms
        self.img_fps = img_fps
        self.txt_fps = txt_fps
        self.img_size = img_size

    def __getitem__(self, idx):
        img_fp = self.img_fps[idx]
        txt_fp = self.txt_fps[idx]
        X, mask = read_img_mask(img_fp, txt_fp)

        if self.transforms is not None:
            transform = self.transforms(image=X, mask=mask)
            X, mask = transform["image"], transform["mask"]

        return X, mask

    def __len__(self):
        return len(self.img_fps)

if __name__ == '__main__':
    from pathlib import Path
    from Transforms import  get_transform
    
    def turn_off_ticks(ax):
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        ax.set_xticks([])
        ax.set_yticks([])        
    
    img_fps = sorted(glob.glob(os.path.join('DataDoorDirectionLabeled', 'train', 'images' , "*.jpg")))
    txt_fps = sorted(glob.glob(os.path.join('DataDoorDirectionLabeled', 'train', 'labels' , "*.txt")))
    print("img_fps", len(img_fps))
    print("txt_fps", len(txt_fps))
    
    img_size = (128, 128)
    dataset_train = DoorDataset(img_fps, txt_fps, img_size, get_transform(train=False, img_size=img_size))
    print("dataset_train", len(dataset_train))
    
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    axes = axes.flatten()
    for ax_idx, img_idx in enumerate(np.random.choice(range(len(dataset_train)), 16)):
        img, mask = dataset_train[img_idx]
        print("type(img)", type(img))
        print("type(mask)", type(mask))
        
        X_draw = (img.cpu().detach().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        X_mask = (mask.cpu().detach().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        print("X_draw.shape", X_draw.shape)
        print("X_mask.shape", X_mask.shape)
        
        # if np.sum(X_mask[0]) > 0:
        #     contours, hierarchy = cv2.findContours(image=X_mask[0], mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        #     cv2.drawContours(image=X_draw, contours=contours, contourIdx=-1, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        axes[ax_idx * 2].imshow(X_draw, cmap='gray')
        axes[ax_idx * 2 + 1].imshow(X_mask, cmap='gray')
        axes[ax_idx * 2].set_title(str(img_idx))
        turn_off_ticks(axes[ax_idx * 2])
        turn_off_ticks(axes[ax_idx * 2 + 1])
        

    plt.tight_layout()
    Path("temp").mkdir(exist_ok=True)
    plt.savefig(os.path.join('temp', 'DatasetPreviewDsiffSample.png'))
    print("write file to", os.path.abspath(os.path.join('temp', 'DatasetPreviewDsiffSample.png')))
