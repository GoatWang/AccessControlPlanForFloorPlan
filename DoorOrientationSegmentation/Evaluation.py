if __name__ == '__main__':
    import os 
    import cv2
    import glob
    import torch
    import numpy as np
    from Model import Unet
    from Metrics import cal_iou
    from Dataset import DoorDataset
    from matplotlib import pyplot as plt
    from Transforms import  get_transform
    
    def turn_off_ticks(ax):
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        ax.set_xticks([])
        ax.set_yticks([])        
    

    params = {
        "device": "cuda", # cpu, cuda
        "input_size":(128, 128), 
        "batch_size": 4, 
        "num_workers": 4, 
    }
    device = torch.device(params['device']) if torch.cuda.is_available() else torch.device('cpu')
    # model_fp = 'models/unet/2024-03-10-01-55-52/107_0.015661_0.9188_0.0734_0.6793.pt'
    model_fp = 'models/unet/2024-03-10-01-55-52/405_0.014373_0.9156_0.5066_0.6576.pt'
    model = Unet(params['input_size'], device=device)
    model.load_state_dict(torch.load(model_fp)['state_dict'])
    model.to(device)
    model.eval()

    img_fps_valid = sorted(glob.glob(os.path.join('DataDoorDirectionLabeled', 'valid', 'images' , "*.jpg")))
    txt_fps_valid = sorted(glob.glob(os.path.join('DataDoorDirectionLabeled', 'valid', 'labels' , "*.txt")))
    dataset_valid = DoorDataset(img_fps_valid, txt_fps_valid, params['input_size'], get_transform(train=False, img_size=params['input_size']))

    # dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=params['batch_size'], shuffle=False, num_workers=params['num_workers'])
    
    transform = get_transform(True, params['input_size'])
    fig, axes = plt.subplots(5, 8, figsize=(16, 8))
    axes = axes.flatten()
    
    for ax_idx, img_idx in enumerate(np.random.choice(range(len(dataset_valid)), 40)):
        img, mask = dataset_valid[img_idx]
        Y_pred = model(img[None, :, :, :].to(device))[0] > 0.5
        
        X_draw = (img.cpu().detach().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        X_draw = cv2.cvtColor(X_draw, cv2.COLOR_GRAY2BGR)
        Y_true = (mask.cpu().detach().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        Y_pred = (Y_pred.cpu().detach().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        
        # label (blue)
        contours, hierarchy = cv2.findContours(image=Y_true[:, :, 0], mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        
        cv2.drawContours(image=X_draw, contours=contours, contourIdx=-1, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

        # pred (red)
        contours, hierarchy = cv2.findContours(image=Y_pred[:, :, 0], mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        biggest_contour = max(contours, key=cv2.contourArea)     
        cv2.drawContours(image=X_draw, contours=[biggest_contour], contourIdx=-1, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
            
        axes[ax_idx].imshow(X_draw, vmin=0, vmax=1)
        turn_off_ticks(axes[ax_idx])
        
    plt.tight_layout()
    plt.savefig(os.path.join('temp', 'Evaluation.png'))
    print("write file to", os.path.abspath(os.path.join('temp', 'Evaluation.png')))    
    
    
    