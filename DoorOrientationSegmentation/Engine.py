import torch
import numpy as np

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    losses_epoch = []
    for batch, (X, masks) in enumerate(dataloader):
        X, masks_true = X.to(device), masks.to(device)

        # Compute prediction error
        masks_pred = model(X)
        loss = loss_fn(masks_pred, masks_true)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward() # generate grad foreach parameters
        optimizer.step() # apply grad to parameters

        if batch % 30 == 0:
            loss, current = loss, batch * len(X)
            print(f"loss: {loss:>7f}  [{current:d}/{size:d}]")
        losses_epoch.append(loss.item())
    losses_epoch = np.mean(losses_epoch)
    return losses_epoch

def valid(dataloader, model, loss_fn, device, metric_fns=None):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    metrics_dict = {fn.__name__:0 for fn in metric_fns}
    with torch.no_grad():
        for batch, (X, masks) in enumerate(dataloader):
            X, masks_true = X.to(device), masks.to(device)
            masks_pred = model(X)
            test_loss += loss_fn(masks_pred, masks_true).item()
            for fn in metric_fns:
                metrics_dict[fn.__name__] += fn(masks_pred, masks_true)
    test_loss /= num_batches
    metrics_dict = {fn.__name__:(metrics_dict[fn.__name__]/num_batches).item() for fn in metric_fns}
    return test_loss, metrics_dict

# def predict(model, X, device):
#     X = X.to(device)
#     Y_pred = torch.sigmoid(model(X))
#     Y_pred = Y_pred.cpu().numpy()
#     return Y_pred