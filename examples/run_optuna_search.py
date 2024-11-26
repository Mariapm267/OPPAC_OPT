import sys

# torch
import torch
import torch.nn as nn

# optuna
import optuna

#from this repo
sys.path.append('../')
import regressor_utils   
import regressor

version = 'V2Steps'
output_size = 2 if '1' in version else 12 if '2' in version else None

def objective(trial):
    ht_train_loader = regressor_utils.get_dataloader(file = '../../processed_datasets' + '_2steps' + '/data_tuning_train.pickle', nsteps=2, subsample = 0.01)
    ht_val_loader   = regressor_utils.get_dataloader(file = '../../processed_datasets' + '_2steps' + '/data_tuning_val.pickle', nsteps=2, subsample = 0.01)

    hidden_size =  trial.suggest_categorical('hidden_size', [24, 48, 96, 128, 256, 512, 1024])
    nlayers = trial.suggest_int('nlayers', 2, 20)
    lr = trial.suggest_categorical('lr', [1e-6, 1e-5, 1e-4, 1e-3, 1e-2])

    dropout = 0   #trial.suggest_categorical('dropout', [0, 0.1, 0.2, 0.3, 0.4, 0.5])
    optimizer_name = 'Adam'   #trial.suggest_categorical('optimizer', ['Adam', 'NAdam', 'Adamax'])
    step_size_scheduler = 1   #trial.suggest_categorical('step_size_scheduler', [1, 2, 3, 4, 5])
    gamma_scheduler = 0.999   #trial.suggest_categorical('gamma_scheduler', [0.9, 0.99, 0.999])
    use_batch_norm = True     #trial.suggest_categorical('use_batch_norm', [True, False])
    activation_function = nn.LeakyReLU    #trial.suggest_categorical('activation_function', [nn.ReLU, nn.SELU, nn.ELU, nn.LeakyReLU])

    model = regressor.NeuralNetwork(nlayers=nlayers, hidden_size=hidden_size, dropout=dropout, input_size=4, output_size=output_size, act_fun=activation_function, use_batch_norm=use_batch_norm)
    criterion = nn.MSELoss()
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size_scheduler, gamma=gamma_scheduler)
    early_stopper = regressor_utils.EarlyStopper(patience=10, min_delta=0.0001)
    n_epochs = 200
    for epoch in range(n_epochs):
        train_loss = regressor.train_loop(ht_train_loader, model, criterion, optimizer, scheduler)
        val_loss = regressor.validation_loop(ht_val_loader, model, criterion)
        trial.report(val_loss, epoch)
        
        if trial.should_prune():                        # prune unpromising trials
            raise optuna.exceptions.TrialPruned()
        
        if early_stopper.early_stop(val_loss): 
            print(f'Early stopping on epoch {epoch}')
            break
        
    return val_loss


def print_hyperparameters_and_best(study, trial):
    print(f"Trial {trial.number}: {trial.params}")
    best_trial = study.best_trial
    print(f"Mejor valor hasta ahora: {best_trial.value}")
    print(f"Parámetros del mejor trial hasta ahora: {best_trial.params}")


def main(n_trials):
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(sampler=sampler, direction='minimize')
    study.optimize(objective, n_trials=n_trials, n_jobs = 8, callbacks=[print_hyperparameters_and_best])


    print("Best trial:")
    best_trial = study.best_trial

    print("Value: ", best_trial.value)
    
    with open(f'optuna_log_{version}.txt', 'w') as f:
        f.write('Best trial:\n')
        f.write(f'N trials: {n_trials}\n')
        f.write(f'  Value: {best_trial.value}\n')
        f.write('  Params:\n')
        for key, value in best_trial.params.items():
            f.write(f'    {key}: {value}\n')
    
n_trials = 1000
main(n_trials)
