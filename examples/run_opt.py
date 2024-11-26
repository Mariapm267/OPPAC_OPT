import torch
import sys
import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# From repo
sys.path.append('../')
import passive_generator 
import optimization
from core import DEVICE
from alpha_batch import AlphaBatch
from loss import BeamPositionLoss
 
# load pre-trained reconstruction model in eval mode

version = '1step'
steps = 1 if '1' in version else 2 
model_path = f'../models/Model{version}.pt'

model= torch.load(model_path)                  
model.eval()

# use generator to generate a batch of beam possitions
batch_size = 10000
generator = passive_generator.RandomBeamPositionGenerator()
alpha_batch = AlphaBatch(generator.generate_set(batch_size))


def plot_opt(epochs, all_loss_values, all_p_values, col_length_values, fontsize = 14):
        
        colors = sns.color_palette("rocket", n_colors = len(all_loss_values))
        plt.figure(figsize = (12, 6))
        #plt.suptitle('Optimization for several initial configurations', fontsize = fontsize)
        plt.subplot(1, 2, 1)
        for i in range(len(all_loss_values)):
            plt.plot(range(epochs), np.sqrt(all_loss_values[i]) / 1000, color = colors[i])
        plt.xlabel('Epoch', fontsize = fontsize)
        plt.ylabel('Loss (RMSE) [cm]', fontsize = fontsize)

        plt.subplot(1, 2, 2)
        for i in range(len(all_p_values)):
            plt.plot(range(epochs), all_p_values[i], color = colors[i], label = f' d = {col_length_values[i]:.2f}')
            plt.legend()
        plt.xlabel('Epoch', fontsize = fontsize)
        plt.ylabel('Pressure [Torr]', fontsize = fontsize)

        plt.tight_layout()
        plt.savefig('../../figs_opt/optimization_graph_fixed_d.pdf', dpi=400)


def opt_fixed_d(col_length_values, epochs = 2000):
    p_final_values = []; d_final_values = []
    all_loss_values = []; all_p_values = []; all_d_values = []
    N = len(col_length_values)
    for i in range(N):
        pressure = torch.tensor(random.uniform(10,50), dtype=torch.float32)
        collimator_length = torch.tensor(col_length_values[i], dtype=torch.float32)
        VolumeWrapper = optimization.AbsVolumeWrapper(pressure, collimator_length, alpha_batch, model,  lr=0.1, epochs=epochs, fix_collimator_length = True,  device=DEVICE)
        loss_values, p_values, d_values = VolumeWrapper.fit(steps = steps)
        p_final_values.append(p_values[-1]); d_final_values.append(d_values[-1])
        all_loss_values.append(loss_values)
        all_p_values.append(p_values)
        all_d_values.append(d_values)
    plot_opt(epochs, all_loss_values, all_p_values, col_length_values)


def run_grid(col_length_values_grid, p_values_grid, epochs = 1500):
    p_final_values = []; d_final_values = []
    all_loss_values = []; all_p_values = []; all_d_values = []
    combined_values = []
    d, p = np.meshgrid(col_length_values_grid, p_values_grid)
    d_flat = d.flatten()
    p_flat = p.flatten()
    for d, p in zip(d_flat, p_flat):
        print(d, p)
        pressure = torch.tensor(p, dtype=torch.float32)
        collimator_length = torch.tensor(d, dtype=torch.float32)
        VolumeWrapper = optimization.AbsVolumeWrapper(pressure, collimator_length, alpha_batch, model,  lr = 0.1, epochs = epochs, fix_collimator_length = False,  device = DEVICE)
        loss_values, p_values, d_values = VolumeWrapper.fit(steps = steps)
        p_final_values.append(p_values[-1]); d_final_values.append(d_values[-1])
        all_loss_values.append(loss_values)
        all_p_values.append(p_values)
        all_d_values.append(d_values)
        iteration_values = np.array([loss_values, p_values, d_values]).T
        combined_values.append(iteration_values)
        
    df = pd.DataFrame({'iteration_values': combined_values})

    # Save DataFrame to a file
    filename = 'optimization_results.pickle'
    df.to_pickle(filename)
    print(f"Saved results to {filename}")
    VolumeWrapper.plot_opt(all_loss_values, all_p_values, all_d_values, filename = 'optimization_graph_grid')
    
    
    
def opt_random(N, epochs = 1500):
    p_final_values = []; d_final_values = []
    all_loss_values = []; all_p_values = []; all_d_values = []
    combined_values = []
    for _ in range(N):
        pressure = torch.tensor(random.uniform(10, 50), dtype=torch.float32)
        collimator_length = torch.tensor(random.uniform(5, 50), dtype=torch.float32)
        loss_fun = BeamPositionLoss()
        VolumeWrapper = optimization.AbsVolumeWrapper(pressure, collimator_length, alpha_batch, model,  lr=0.1, epochs=epochs, fix_collimator_length = False,  device=DEVICE, loss_fun = loss_fun)
        loss_values, p_values, d_values = VolumeWrapper.fit(steps = steps)
        p_final_values.append(p_values[-1]); d_final_values.append(d_values[-1])
        all_loss_values.append( np.sqrt(np.array(loss_values)) / 1000)
        all_p_values.append(np.array(p_values))
        all_d_values.append(np.array(d_values))
        iteration_values = np.array([loss_values, p_values, d_values]).T
        combined_values.append(iteration_values)
        
    df = pd.DataFrame({'iteration_values': combined_values})

    # Save DataFrame to a file
    filename = 'optimization_results.pickle'
    df.to_pickle(filename)
    print(f"Saved results to {filename}")

    VolumeWrapper.plot_opt(all_loss_values, all_p_values, all_d_values, filename = 'optimization_graph_random')
    #VolumeWrapper.plot_3D(all_loss_values, all_p_values, all_d_values, fontsize = 14)
    
    
    
def main(what_to_run):
        
    if what_to_run == 'random':
        opt_random(N = 1, epochs = 1500)

    if what_to_run == 'fixed_d':
        col_length_values = np.linspace(5, 50, 15)
        print(col_length_values)
        opt_fixed_d(col_length_values)
        
    if what_to_run == 'grid':
        col_length_values_grid = np.linspace(5, 50, 10)
        p_values_grid = np.linspace(10, 50, 10)   
        run_grid(col_length_values_grid, p_values_grid)
        
if __name__ == "__main__":
    what_to_run = 'random'
    main(what_to_run=what_to_run)
