
from torch import tensor, Dataset, float32
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
        features = tensor(self.data.loc[idx, self.features].values, dtype=float32)
        targets = tensor(self.data.loc[idx, self.targets].values, dtype=float32)
        return targets, features

    def __len__(self): 
        return len(self.data)