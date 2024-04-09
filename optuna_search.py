import torch
import torch.nn as nn

import optuna

#from this repo
import regressor_utils   
import regressor


def objective(trial):
    _, _, _, ht_train_loader, ht_val_loader = regressor_utils.get_dataloaders(file = 'processed_data.pickle')
    
    hidden_size_trial = trial.suggest_categorical('hidden_size', [64, 128, 256, 512, 1024])
    nlayer_trial = trial.suggest_int('nlayers', 3, 10)
    lr_trial = trial.suggest_float('lr', 1e-4, 1e-1)

    model = regressor.NeuralNetwork(nlayers=nlayer_trial, hidden_size=hidden_size_trial, input_size = 2, output_size = 4)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_trial)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9) 
    
    early_stopper = regressor_utils.EarlyStopper(patience = 10, min_delta=10)
    n_epochs = 10
    for epoch in range(n_epochs):
        train_loss = regressor.train_loop(ht_train_loader, model, criterion, optimizer, scheduler)
        val_loss = regressor.validation_loop(ht_val_loader, model, criterion)
        trial.report(val_loss, epoch)
        
        if trial.should_prune():                        # prune unpromising trials
            raise optuna.exceptions.TrialPruned()
        
        if early_stopper.early_stop(val_loss): 
            print(f"Early stopping on epoch {epoch}")
            break
        
    return val_loss



def main(n_trials):
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials)
    
    pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    
    # save best params in a txt
    with open('optuna_log.txt', 'w') as f:
        f.write('Best trial:\n')
        best_trial = study.best_trial
        f.write(f'N trials: {n_trials}')
        f.write(f'  Value: {best_trial.value}\n')
        f.write('  Params:\n')
        for key, value in best_trial.params.items():
            f.write(f'    {key}: {value}\n')
    
n_trials = 100
main(n_trials)