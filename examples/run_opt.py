import torch

# Basic stuff
import sys
import random
import matplotlib.pyplot as plt
import os

# From this repo
sys.path.append('../')
import optimization
from core import DEVICE
from alpha_batch import AlphaBatch

# load pre-trained reconstruction model in eval mode
model_path = '../models/Model.pt'
model= torch.load(model_path)                  
model.eval()

# generate an alpha batch (positions of random impining particles)
batch_size = 10000
alpha_batch = AlphaBatch(batch_size=batch_size)

# Initial values of pressure and collimator length
pressure = torch.tensor(random.uniform(10,50), dtype=torch.float32)
collimator_length = torch.tensor(random.uniform(5,50), dtype=torch.float32)

VolumeWrapper = optimization.AbsVolumeWrapper(pressure, collimator_length, alpha_batch, model,  lr=0.1, epochs=1000,  device=DEVICE)

# fit
loss_values, p_values, d_values = VolumeWrapper.fit()


plot = True
if plot:
    figs_folder = '../figs/'
    if not os.path.exists(figs_folder):
      os.makedirs(figs_folder)
    VolumeWrapper.plot_loss(loss_values, path = '../figs/loss_opt.pdf')
    VolumeWrapper.plot_pressure(p_values, path = '../figs/pressure_opt.pdf')
    VolumeWrapper.plot_col_length(d_values, path = '../figs/length_opt.pdf')
