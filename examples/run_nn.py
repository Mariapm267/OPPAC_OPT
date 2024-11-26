#torch
import torch.nn as nn
import torch

#basic stuff
import numpy as np
from sklearn.metrics import root_mean_squared_error
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt

#from this repo
import sys
sys.path.append('../')
import regressor_utils   
import regressor


version = '1step'
folder = f'/scratch04/maria.pereira/TFM/processed_datasets_{version}'
results_folder = f'../../results_{version}/'

batch_size = 1000
epochs = 100
lr = 1e-4
n_layers = 6
hidden_size = 1024
gamma_scheduler = 0.999
step_size_scheduler = 1
do = 0.
activation_fun = nn.LeakyReLU
batch_norm = True
output_size = 2 if '1' in version else 12 if '2' in version else None


def get_and_save_results_1step(model, test_loader, filename = results_folder + 'results' + version + '.pickle'):
    pred_pos = []
    true_pos = []
    reco_pos = []
    model.eval()      
    with torch.no_grad():
        for (y_test, X_test) in test_loader:
            pred = model(X_test)
            pred_pos.extend(pred.numpy())     
            reco_pos.extend(y_test.numpy())
            true_pos.extend(X_test[0:, 2:].numpy())
    rmse_x = root_mean_squared_error([t[0] for t in reco_pos], [p[0] for p in pred_pos])
    rmse_y = root_mean_squared_error([t[1] for t in reco_pos], [p[1] for p in pred_pos])

    print(f'RMSE for x: {rmse_x/1000} cm')
    print(f'RMSE for y: {rmse_y/1000} cm')
    
    df = pd.DataFrame({
        'true_pos': true_pos,
        'pred_pos': pred_pos,
        'reco_pos': reco_pos
    })
    
    with open(filename, 'wb') as f:
        pickle.dump(df, f)
        

def get_and_save_results_2steps(model, test_loader, filename = results_folder + 'results' + version + '.pickle'):
    model.eval()   
    pred_stats = []
    true_stats = []  
    true_pos = []
    with torch.no_grad():
        for (y_test, X_test) in test_loader:
            pred = model(X_test)
            pred_stats.extend(pred.numpy())     
            true_stats.extend(y_test.numpy())
            true_pos.extend(X_test[0:, 2:].numpy())
    
    df = pd.DataFrame({
        'pred_stats': pred_stats,
        'true_stats': true_stats,
        'true_pos': true_pos
    })

    with open(filename, 'wb') as f:
        pickle.dump(df, f)


def save_loss(loss_train, loss_val):
    df = pd.DataFrame({
        'loss_train': loss_train,
        'loss_val': loss_val
    })
    with open(results_folder + 'loss' + version + '.pickle', 'wb') as f:
        pickle.dump(df, f)


def main():

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    train_loader = regressor_utils.get_dataloader(file = folder + '/data_tuning_train.pickle', batch_size = batch_size, nsteps=2, subsample = 0.1)
    val_loader = regressor_utils.get_dataloader(file = folder + '/data_tuning_val.pickle', batch_size = batch_size, nsteps=2, subsample = 0.1)
    test_loader = regressor_utils.get_dataloader(file = folder + '/data_remaining.pickle', batch_size = batch_size, nsteps=2, subsample = 0.001)

    model = regressor.NeuralNetwork(nlayers=n_layers, hidden_size=hidden_size, dropout = do, input_size = 4, output_size = output_size, act_fun = activation_fun, use_batch_norm=batch_norm)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=gamma_scheduler, step_size=step_size_scheduler) 
    
    loss_fn = nn.MSELoss()

    train_losses, val_losses = regressor.fit(train_loader, val_loader, model, epochs, loss_fn, optimizer, scheduler, early_stopper = regressor_utils.EarlyStopper(patience=10, min_delta=0.01), use_tqdm=True)

    plt.figure(figsize = (8, 8))
    plt.plot(range(len(train_losses)), train_losses, label = 'Training', color = 'red')
    plt.plot(range(len(val_losses)), val_losses, label = 'Validation', color = 'blue')
    plt.legend()
    plt.savefig(results_folder + 'loss.pdf', dpi = 400)

    PATH = f'../models/Model{version}.pt'     
    torch.save(model, PATH)
    if '1' in version:
        get_and_save_results_1step(model, test_loader)
    elif '2' in version:
        get_and_save_results_2steps(model, test_loader)
        
    save_loss(train_losses, val_losses)
    
    
if __name__ == '__main__':
   main()
