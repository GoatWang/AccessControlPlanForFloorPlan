import os 
import cv2
import sys
import glob
import torch
import numpy as np
from matplotlib import pyplot as plt
sys.path.append(os.path.dirname(__file__))
from Transforms import get_transform

def find_door(mask):
    contours, hierarchy = cv2.findContours(image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0 :
        biggest_contour = max(contours, key=cv2.contourArea)
        xs, ys = biggest_contour.T
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        return x1, y1, x2, y2
    else:
        return 0, 0, 0, 0
    
def predict(model, images):
    """
    Returen
    ----------
    pred: 255 for door type: uint8
    """
    device = 'cuda' if torch.cuda.is_available() else torch.device('cpu')
    size_ori = []
    images_trans = []
    for image in images:
        assert len(image.shape) == 2, "image should be in gray scale"
        if image.dtype == np.uint8:
            image = (image/255).astype(np.float32)

        assert image.dtype == np.float32, "image should be in uint8 or float32 type"
        size_ori.append(image.shape[:2][::-1])
        transform = get_transform(False, img_size=(128, 128))
        image_trans = transform(image=image)['image']
        images_trans.append(image_trans.to(device))
        
    preds = model(torch.stack(images_trans)) > 0.5
    preds = np.squeeze(preds.detach().cpu().numpy(), axis=1)
    preds = (preds * 255).astype(np.uint8)
    
    mask_preds = [cv2.resize(pred, size) for pred, size in zip(preds, size_ori)]
    box_preds = [find_door(pred) for pred in mask_preds]
    return mask_preds, box_preds

    
if __name__ == '__main__':
    from pathlib import Path
    from Dataset import read_img_mask
    from matplotlib import pyplot as plt
    import matplotlib.patches as patches
    # model = load_model('models/unet/2024-03-10-01-55-52/107_0.015661_0.9188_0.0734_0.6793.pt')
    # model = load_model('models/unet/2024-03-10-01-55-52/405_0.014373_0.9156_0.5066_0.6576.pt')
    model = torch.jit.load("weights/best.pt")

    img_fps_valid = sorted(glob.glob(os.path.join('DataDoorDirectionLabeled', 'valid', 'images' , "*.jpg")))
    txt_fps_valid = sorted(glob.glob(os.path.join('DataDoorDirectionLabeled', 'valid', 'labels' , "*.txt")))
    
    image = cv2.cvtColor(cv2.imread(img_fps_valid[0]), cv2.COLOR_BGR2GRAY)
    mask_preds, box_preds = predict(model, [image])
    mask_pred, box_pred = mask_preds[0], box_preds[0]
    
    x1, y1, x2, y2 = box_pred
    Path('temp').mkdir(exist_ok=True)
    X, y_mask_true = read_img_mask(img_fps_valid[0], txt_fps_valid[0])
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3))
    ax1.imshow(image, cmap='gray')
    ax1.set_title("source image")
    ax1.add_patch(patches.Rectangle(xy=(x1, y1), 
                             width=x2-x1, 
                             height=y2-y1,
                             linewidth=1,
                             edgecolor='blue', 
                             facecolor='none'))    
    
    
    ax2.imshow(mask_pred, cmap='gray')
    ax2.set_title("prediction")
    ax3.imshow(y_mask_true, cmap='gray')
    ax3.set_title("label")
    plt.tight_layout()
    plt.savefig(os.path.join("temp", "predict.png"))
    print("prediction saved to", os.path.join("temp", "predict.jpg"))
    