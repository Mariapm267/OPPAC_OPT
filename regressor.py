import torch.nn as nn
import numpy as np
import torch
from tqdm import tqdm


class NeuralNetwork(nn.Module):
    def __init__(self, nlayers, hidden_size, dropout, input_size, output_size):
        super().__init__()
        layers = []
        for i in range(nlayers - 1):
            layers.append(nn.Linear(input_size if i == 0 else hidden_size, hidden_size))
            #layers.append(nn.BatchNorm1d(hidden_size))  # batch normalization
            layers.append(nn.ReLU())
            #layers.append(nn.Dropout(p=dropout))  # dropout
            
            
        layers.append(nn.Linear(hidden_size, output_size))
        self.linear_relu_stack = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x




def train_loop(dataloader, model, loss_fn, optimizer, scheduler):
    losses = []
    model.train()   # (training mode)
    with tqdm(total=len(dataloader), desc='Training', unit='batch') as pbar:
        for i, (y, X) in enumerate(dataloader):   # for each batch
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X)
            loss = loss_fn(outputs, y)

            # Backpropagation
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            
            pbar.set_postfix({'Loss': loss.item()})
            pbar.update(1)
            
    scheduler.step()     # scheduler update at each epoch
    return np.mean(losses)




def validation_loop(dataloader, model, loss_fn):
    losses = []
    model.eval()     # (evaluation mode) 
    
    with tqdm(total=len(dataloader), desc='Validation', unit='batch') as pbar:
        with torch.no_grad():
            for y, X in dataloader:
                pred = model(X)
                loss = loss_fn(pred, y).item()
                losses.append(loss)
                pbar.set_postfix({'Loss': loss})
                pbar.update(1)
                
    return np.mean(losses)



    
