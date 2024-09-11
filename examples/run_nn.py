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


def load_best_params(filename='best_params.pickle'):
    '''
    Loads the hyperparameters obtained with optuna
    '''
    with open(filename, 'rb') as f:
        best_params = pickle.load(f)
    return best_params


def evaluate_model(model, test_loader):
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


def save_loss(loss_train, loss_val, filename = '../../loss.pickle'):
    df2 = pd.DataFrame({
        'loss_train': loss_train,
        'loss_val': loss_val
    })
    with open('loss.pickle', 'wb') as f:
        pickle.dump(df2, f)

    
def save_results(true_pos, pred_pos, reco_pos,  filename = '../../pos_results.pickle'):
    df = pd.DataFrame({
        'true_pos': true_pos,
        'pred_pos': pred_pos,
        'reco_pos': reco_pos
    })
    
    with open(filename, 'wb') as f:
        pickle.dump(df, f)
        

def main():

    # Load data
    folder = '../../processed_datasets'
    train_loader = regressor_utils.get_dataloader(file=f'{folder}/data_train.pickle')
    val_loader = regressor_utils.get_dataloader(file=f'{folder}/data_val.pickle')
    
    eval_mode = True   # load evaluation dataset or not
    if eval_mode:
        test_loader = regressor_utils.get_dataloader(file=f'{folder}/data_remaining.pickle')

    # Model definition
    best_params = load_best_params()

    nsteps = 1
    lr = 0.0352
    n_layers = 3
    hidden_size = 64
    gamma_scheduler = best_params['gamma_scheduler']
    step_size_scheduler = best_params['step_size_scheduler']
    do =  best_params['do']
    activation_fun =  best_params['activation_fun']
    batch_norm =best_params['use_batch_norm']

    if nsteps == 1:
        output_size = 2
    elif nsteps == 2:
        output_size = 12

    model = regressor.NeuralNetwork(
        nlayers=n_layers, hidden_size=hidden_size, dropout=do,
        input_size=4, output_size=output_size, act_fun=activation_fun,
        use_batch_norm=batch_norm
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adamax(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=gamma_scheduler, step_size=step_size_scheduler)
    epochs = 100
    loss_fn = nn.MSELoss()

    # Training
    train_losses, val_losses = regressor.fit(train_loader, val_loader, model, epochs, loss_fn, optimizer, scheduler, use_tqdm=True)


    # Save model
    PATH = '../models/Model.pt'
    torch.save(model, PATH)

    print('Saving loss...')
    save_loss(train_losses, val_losses, filename='../../loss.pickle')

    if eval_mode:
        print('Evaluating on test set...')
        true_pos, pred_pos, reco_pos = evaluate_model(model, test_loader)
        print('Saving results...')
        save_results(true_pos, pred_pos, reco_pos, filename='position_results.pickle')

if __name__ == "__main__":
    main()
