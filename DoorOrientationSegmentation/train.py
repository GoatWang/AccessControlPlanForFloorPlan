import os
import glob
import torch
from Model import Unet
from Loss import CELoss
from pathlib import Path
from datetime import datetime
from Engine import train, valid
from Dataset import DoorDataset
from Transforms import get_transform
from Metrics import IoU, SoftIoU, AccuracyAtIoU

params = {
    "model": "unet",
    "device": "cuda", # cuda
    "input_size":(128, 128), 
    "lr": 0.001,
    "batch_size": 8,
    "num_workers": 8,
    "num_epochs": 2000,
}
model_dir = os.path.join('models', params['model'])
Path(model_dir).mkdir(parents=True, exist_ok=True)

device = torch.device(params['device']) if torch.cuda.is_available() else torch.device('cpu')

img_fps_train = sorted(glob.glob(os.path.join('DataDoorDirectionLabeled', 'train', 'images' , "*.jpg")))
txt_fps_train = sorted(glob.glob(os.path.join('DataDoorDirectionLabeled', 'train', 'labels' , "*.txt")))
img_fps_valid = sorted(glob.glob(os.path.join('DataDoorDirectionLabeled', 'valid', 'images' , "*.jpg")))
txt_fps_valid = sorted(glob.glob(os.path.join('DataDoorDirectionLabeled', 'valid', 'labels' , "*.txt")))

dataset_train = DoorDataset(img_fps_train, txt_fps_train, params['input_size'], get_transform(train=True, img_size=params['input_size']))
dataset_valid = DoorDataset(img_fps_valid, txt_fps_valid, params['input_size'], get_transform(train=False, img_size=params['input_size']))

# define training and validation data loaders
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=params['batch_size'], shuffle=True, num_workers=params['num_workers'])
dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=params['batch_size'], shuffle=False, num_workers=params['num_workers'])

# get model
model = Unet(input_size=params['input_size'], device=device)
model.to(device)

# construct model and an optimizer
learning_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(learning_params, lr=params['lr'])
loss_fn = CELoss(device) # CELoss(device), SoftDiceLoss()
metric_fns = [IoU(threshold=0.5), SoftIoU(), AccuracyAtIoU(threshold=0.5, iou_threshold=0.90)]
monitors = [metric_fns[i].__name__ for i in range(3)]
modes = ['maximize'] * 8 # or minimize

# callback
model_dt_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
current_model_dir = os.path.join(model_dir, model_dt_str)
Path(current_model_dir).mkdir(parents=True, exist_ok=True)
csv_logger_fp = os.path.join(current_model_dir, "model_history_log.csv")
with open(csv_logger_fp, 'w') as f:
    metric_names = ["epoch", "train_loss", "valid_loss"] + [fn.__name__ for fn in metric_fns]
    f.write(", ".join(metric_names) + "\n")

# let's train it 
best_scores = [(-10**6 if mode == 'maximize' else 10**6) for mode in modes]
for epoch in range(params['num_epochs']):
    # train for one epoch, printing every 10 iterations
    train_loss = train(dataloader_train, model, loss_fn, optimizer, device)

    # evaluate on the test dataset
    valid_loss, metrics_dict = valid(dataloader_valid, model, loss_fn, device, metric_fns)
    scores = [metrics_dict[monitor] for monitor in monitors]
    metric_scores = [f"{metrics_dict[fn.__name__]:.4f}" for fn in metric_fns]
    checkpoint = {
        "model": model,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": params,
        "epoch": epoch,
    }

    log_info = [f"{epoch:4d}"] + [f"{metric:.6f}" for metric in [train_loss, valid_loss]] + metric_scores
    print(", ".join([mn + ": "+ m for mn, m in zip(metric_names, log_info)]) + "\n")

    scores_imrpoved = [((mode=='maximize') and (score>=best_score)) or 
                       ((mode=='minimize') and (score<=best_score)) 
                       for score, best_score, mode in zip(scores, best_scores, modes)]
    if any(scores_imrpoved):
        model_fp = os.path.join(current_model_dir, f'{epoch:d}_{valid_loss:.6f}_' + "_".join(metric_scores) + '.pt')
        torch.save(checkpoint, model_fp)
        best_scores = [score if score_imrpoved else best_score for score, best_score, score_imrpoved in zip(scores, best_scores, scores_imrpoved)]
        print('model saved:', model_fp)

    with open(csv_logger_fp, 'a') as f:
        f.write(", ".join(log_info) + "\n")



