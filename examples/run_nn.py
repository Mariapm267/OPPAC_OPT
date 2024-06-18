#torch
import torch.nn as nn
import torch

#basic stuff
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd
import pickle

#from this repo
import sys
sys.path.append('../')
import regressor_utils   
import regressor

#########################################################################################


folder = '../../processed_datasets'
train_loader = regressor_utils.get_dataloader(file = f'{folder}/data_train.pickle')
val_loader = regressor_utils.get_dataloader(file = f'{folder}/data_val.pickle')
test_loader = regressor_utils.get_dataloader(file = f'{folder}/data_remaining.pickle')


lr = 0.03520592774471439
n_layers = 3
hidden_size = 64
gamma_scheduler = 0.939346938928145
step_size_scheduler = 3
do = 0
activation_fun = nn.SELU
batch_norm = False

model = regressor.NeuralNetwork(nlayers=n_layers, hidden_size=hidden_size, dropout = do, input_size = 4, output_size = 2, act_fun = activation_fun, use_batch_norm=batch_norm)
criterion = nn.MSELoss()
optimizer = torch.optim.Adamax(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=gamma_scheduler, step_size=step_size_scheduler) 
epochs = 100
loss_fn = nn.MSELoss()

train_losses, val_losses = regressor.fit(train_loader, val_loader, model, epochs, loss_fn, optimizer, scheduler, use_tqdm=True)

PATH = '../models/Model2.pt'     # model after second optuna
torch.save(model, PATH)


def get_pos(model, test_loader):
    pred_pos = []
    true_pos = []
    reco_pos = []
    
    model.eval()       # evaluation mode
    with torch.no_grad():
        for (y_test, X_test) in test_loader:
            pred = model(X_test)
            pred_pos.extend(pred.numpy())     
            reco_pos.extend(y_test.numpy())
            true_pos.extend(X_test[0:, 2:].numpy())

    rmse_x = np.sqrt(mean_squared_error([t[0] for t in reco_pos], [p[0] for p in pred_pos]))
    rmse_y = np.sqrt(mean_squared_error([t[1] for t in reco_pos], [p[1] for p in pred_pos]))

    print(f'RMSE for x: {rmse_x/1000} cm')
    print(f'RMSE for y: {rmse_y/1000} cm')
    return true_pos, pred_pos, reco_pos

def save_results(true_pos, pred_pos, reco_pos, loss_train, loss_val,  filename = '../../test_results.pickle'):
    df = pd.DataFrame({
        'true_pos': true_pos,
        'pred_pos': pred_pos,
        'reco_pos': reco_pos
    })
    
    df2 = pd.DataFrame({
        'loss_train': loss_train,
        'loss_val': loss_val
    })
    
    with open(filename, 'wb') as f:
        pickle.dump(df, f)
        
    with open('loss_3_prime.pickle', 'wb') as f:
        pickle.dump(df2, f)

    return None



print('Evaluate on test set...')
true_pos, pred_pos, reco_pos = get_pos(model, test_loader)

print('Saving results...')
save_results(true_pos, pred_pos, reco_pos, train_losses, val_losses, filename = 'test_results_3_prime.pickle')