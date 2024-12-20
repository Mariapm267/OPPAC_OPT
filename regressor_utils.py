
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import os


class AlphasDataset(Dataset):
    '''
    Dataset class for the regression between simulation inputs and outputs
    '''
    def __init__(self, filename='processed_data.pickle', nsteps=1, subsample = 1):
        self.filename = filename
        self.nsteps = nsteps
        self.data = pd.DataFrame(pd.read_pickle(self.filename)).sample(frac = subsample).reset_index()
        
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
    def __init__(self, patience=10, min_delta=0.001):
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
 
def get_dataloader(file, batch_size = 100, nsteps=1, subsample = 1):
    dataset    = AlphasDataset(file, nsteps=nsteps, subsample = subsample)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
    

def split_data(file, dest_folder = 'split_processed_datasets', version = '1step', seed = 42):
    
    print('Splitting data...')
    data = pd.read_pickle(file)
    data = pd.DataFrame(data)
    print(f'The dataframe has {len(data)} entries')
    
    stratify_vars = ['dcol', 'p', 'x', 'y']   # to mantain proportions

    if 'step' in version:
        nsteps = 1 if '1' in version else 2
        if nsteps == 1:
            targets = ['x_hat', 'y_hat']
            X_train_full, X_remain, y_train_full, y_remain = train_test_split(data.drop(columns=targets), data[targets], test_size=0.9, stratify=data[stratify_vars], random_state=42)
        
        elif nsteps == 2:
            targets = ['Nx1','mux1', 'sigmax1', 'Nx2', 'mux2', 'sigmax2', 'Ny1', 'muy1', 'sigmay1', 'Ny2','muy2', 'sigmay2']
            X_train_full, X_remain, y_train_full, y_remain = train_test_split(data.drop(columns=targets), data[targets], test_size=0.9, stratify=data[stratify_vars], random_state=42)
        
        else:
            print('Steps has to be 1 or 2')
         
    elif 'photon' in version:
        targets = ['nphotons']
        X_train_full, X_remain, y_train_full, y_remain = train_test_split(data.drop(columns=targets), data[targets], test_size=0.95, stratify=data[stratify_vars], random_state=42)

    if nsteps == 1:
        target = ['dcol', 'p', 'x', 'y']
    elif nsteps == 2:
        target = ['Nx1', 'mux1', 'sigmax1', 
                  'Nx2', 'mux2', 'sigmax2', 
                  'Ny1', 'muy1', 'sigmay1', 
                  'Ny2', 'muy2', 'sigmay2']
    else:
        raise ValueError('Number of nsteps not valid (1 or 2)')

    X_train_full, X_remain, y_train_full, y_remain = train_test_split(data.drop(columns=target), data[target], test_size=0.95, stratify=data[stratify_vars], random_state=seed)
    X_tuning, X_train, y_tuning, y_train = train_test_split(X_train_full, y_train_full, test_size=0.95, stratify=X_train_full[stratify_vars], random_state=seed)
    X_tuning_train, X_tuning_val, y_tuning_train, y_tuning_val = train_test_split(X_tuning, y_tuning, test_size=0.50, stratify=X_tuning[stratify_vars], random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.50, stratify=X_train[stratify_vars], random_state=seed)
    	
    data_remaining = pd.concat([X_remain, y_remain], axis=1)
    data_tuning_val = pd.concat([X_tuning_val, y_tuning_val], axis=1)
    data_tuning_train = pd.concat([X_tuning_train, y_tuning_train], axis=1)
    data_val = pd.concat([X_val, y_val], axis=1)
    data_train = pd.concat([X_train, y_train], axis=1)
  
    list_data = [
    ('data_remaining', data_remaining),
    ('data_train', data_train),
    ('data_tuning_train', data_tuning_train),
    ('data_val', data_val),
    ('data_train', data_train),
    ('data_tuning_val', data_tuning_val)
    ]
    
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for name, data in list_data:
        data.reset_index(drop=True, inplace=True)
        with open(f'{dest_folder}/{name}.pickle', 'wb') as f:
            pickle.dump(data.to_dict(), f)

    return None
