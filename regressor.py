import torch.nn as nn
import numpy as np
import torch
from tqdm import tqdm
import regressor_utils

class NeuralNetwork(nn.Module):
    def __init__(self, nlayers, hidden_size, dropout, input_size, output_size, act_fun, use_batch_norm):
        super().__init__()
        layers = []    
        for i in range(nlayers - 1):
            layers.append(nn.Linear(input_size if i == 0 else hidden_size, hidden_size))
            if use_batch_norm == True:
                layers.append(nn.BatchNorm1d(hidden_size))  
            layers.append(act_fun())
            layers.append(nn.Dropout(p=dropout))  
            
            
        layers.append(nn.Linear(hidden_size, output_size))
        self.linear_relu_stack = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x



def train_loop(dataloader, model, loss_fn, optimizer, scheduler, use_tqdm=False):
    losses = []
    model.train()   # (training mode)
    
    if use_tqdm:
        pbar = tqdm(total=len(dataloader), desc='Training', unit='batch')
    
    for i, (y, X) in enumerate(dataloader):   # for each batch
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X)
        loss = loss_fn(outputs, y)

        # Backpropagation
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        
        if use_tqdm:
            pbar.set_postfix({'Loss': loss.item()})
            pbar.update(1)
            
    if use_tqdm:
        pbar.close()
        
    scheduler.step()     # scheduler update at each epoch
    return np.mean(losses)


def validation_loop(dataloader, model, loss_fn, use_tqdm=False):
    losses = []
    model.eval()     # (evaluation mode) 
    
    if use_tqdm:
        pbar = tqdm(total=len(dataloader), desc='Validation', unit='batch')
    
    with torch.no_grad():
        for y, X in dataloader:
            pred = model(X)
            loss = loss_fn(pred, y).item()
            losses.append(loss)
            if use_tqdm:
                pbar.set_postfix({'Loss': loss})
                pbar.update(1)
                
    if use_tqdm:
        pbar.close()
        
    return np.mean(losses)




def fit(train_loader, val_loader, model, epochs, loss_fn, optimizer, scheduler, early_stopper = regressor_utils.EarlyStopper(), use_tqdm=False):
    train_losses=[]
    val_losses=[]
      
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = train_loop(train_loader, model, loss_fn, optimizer, scheduler, use_tqdm)
        val_loss  = validation_loop(val_loader, model, loss_fn, use_tqdm)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print("Avg train loss", train_loss, ", Avg val loss", val_loss, "Current learning rate", scheduler.get_last_lr(), "\n")
        if early_stopper.early_stop(val_loss): 
            print(f"Early stopping on epoch {t}")
            break
        
    print("Done!")
    return train_losses, val_losses
