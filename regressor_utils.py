
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd

class AlphasDataset(Dataset):
    '''
    Dataset class for the regression between simulation inputs and outputs
    '''
    def __init__(self, filename='processed_data.pickle', nsteps=1):
        self.filename = filename
        self.nsteps = nsteps
        self.data = pd.DataFrame(pd.read_pickle(self.filename))  
        
        self.features = ['dcol', 'p', 'x', 'y']
        if nsteps == 1:
            self.targets = ['x_hat', 'y_hat']
        elif nsteps == 2:
            self.targets = ['Nx1', 'mux1', 'sigmax1', 'Nx2', 'mux2', 'sigmax2',
                            'Ny1', 'muy1', 'sigmay1', 'Ny2', 'muy2', 'sigmay2']
        else:
            raise ValueError('Number of nsteps not valid (1 or 2)')

    def __getitem__(self, idx):
        features = torch.tensor(self.data.loc[idx, self.features].values, dtype=torch.float32)
        targets = torch.tensor(self.data.loc[idx, self.targets].values, dtype=torch.float32)
        return targets, features

    def __len__(self): 
        return len(self.data)
    

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
def get_dataloaders(file):
    full_dataset = AlphasDataset(file)
    # split into train, validation and test:
    train_size = 0.7
    val_size = 0.1
    test_size = 0.1
    ht_size = 0.1
    
    torch.manual_seed(42)   
    
    train_dataset, val_dataset, test_dataset, ht_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size, ht_size])
    ht_train, ht_val = torch.utils.data.random_split(ht_dataset, [0.5,0.5])
    batch_size = 50  

    train_loader      = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader        = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader       = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    ht_train_loader   = DataLoader(ht_train, batch_size=batch_size, shuffle=True)
    ht_val_loader   = DataLoader(ht_val, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, ht_train_loader, ht_val_loader
    
    
    
def split_dataset(full_dataset, train_size, test_size):
    ht_size = 1 - train_size - test_size
    torch.manual_seed(42) 
    train_dataset, test_dataset, ht_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size, ht_size])
    return train_dataset, test_dataset, ht_dataset
