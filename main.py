import torch
import numpy as np

from torch.optim import lr_scheduler, RMSprop, Adam, AdamW
from config import Config

from load_data import create_dataloader
from model.writerModel import  ContrastiveLoss, Model, BaseModel, ContrastiveLoss_test

import wandb
import torch.nn as nn

cfg = Config().parse()

train_loader, val_loader = create_dataloader()

def train(epoch, data, data_vl):
    train_loss = [0] * epoch
    # train_accuracy = [0] * epoch
    
    valid_loss = [0] * epoch
    # valid_accuracy = [0] * epoch
    g = 2048
    best_valid = 10000 
    # torch.autograd.set_detect_anomaly(True)
    # with torch.autograd.detect_anomaly():
    for i in range(epoch):
        model.train()
        optimizer.zero_grad()

        for t, (mask_batch, x_batch, y_batch, indices) in enumerate(data):
            x_batch = x_batch/255
            mask_batch, x_batch, y_batch = mask_batch.to(device), x_batch.to(device), y_batch.to(device)
            pred = model(x_batch, mask_batch, indices)
            
            # print(pred)
            # loss = loss_fn(pred, y_batch)
            loss = loss_fn(pred[:, 0, :], pred[:, 1, :], y_batch)
            (loss/g).backward()

            # loss.backward()
            # optimizer.step()
            # optimizer.zero_grad()
            
            if (t + 1) % g == 0:
                optimizer.step()
                optimizer.zero_grad()
                train_loss[i] += loss.item() * x_batch.size(0) * g
                print('train loss :', train_loss[i]) 
                print()
            # train_loss[i] += loss.item() * x_batch.size(0)
            # print(train_loss[i])
        train_loss[i] /= len(data.dataset)
    
        print('Evaluation')
        model.eval()
        with torch.no_grad():
            for t, (mask_batch, x_batch, y_batch, indices) in enumerate(data_vl):
                x_batch = x_batch/255
                mask_batch, x_batch, y_batch = mask_batch.to(device), x_batch.to(device), y_batch.to(device)
                pred = model(x_batch, mask_batch, indices)

                loss = loss_fn(pred[:, 0, :], pred[:, 1, :], y_batch)
                
                print('valid :', loss) 

                valid_loss[i] += loss.item() * x_batch.size(0)

            valid_loss[i] /= len(data_vl.dataset)
        print(f'Epoch {i+1} loss: {train_loss[i]:.4f} val_loss:{valid_loss[i]:.4f}')
        print()
        
        wandb.log({'Train Loss': train_loss[i], 'Valid Loss': valid_loss[i]})
        if valid_loss[i] < best_valid:
            torch.save(model, 'model.pth')
            best_valid = valid_loss[i]           
    return train_loss, valid_loss


model = BaseModel(cfg.batch_size)
# model = Model(cfg.batch_size)

def print_gradients(grad):
    print(torch.sum(grad))
    # print(grad)

# Register hooks to the model parameters
def register_hooks(model):
    for param in model.parameters():
        param.register_hook(print_gradients)

# register_hooks(model)
# model = Model()

wandb.init(
    project="NCVPRIPG",
    name="using valid data",
    config={
        "patch size": cfg.patch_size,
        "epochs": cfg.epochs,
        "batch size": cfg.batch_size,
        "learning_rate": 5e-3,
        "Params,optms": "Transformer"
    }
)

device = torch.device('cuda')
model = torch.load('model.pth', map_location=device)
model = model.to(device)

# loss_fn = ContrastiveLoss(alpha=0.5, beta=0.5, margin=2).to(device)
loss_fn = ContrastiveLoss_test().to(device)

#loss_fn = nn.BCEWithLogitsLoss()

optimizer = AdamW(model.parameters(), lr=4e-4, betas=(0.9, 0.999))
#optimizer = RMSprop(model.parameters(), lr=1e-5, eps=1e-8, weight_decay=5e-4, momentum=0.9)
#scheduler = lr_scheduler.StepLR(optimizer, 5, 0.1)
epochs = cfg.epochs

train_loss, val_loss = train(epochs, train_loader, val_loader)


#torch.save(model, 'model.pth')
del model
model = torch.load('model.pth', map_location=device)
