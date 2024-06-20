import torch
import sys
import random
import matplotlib.pyplot as plt


# From repo
sys.path.append('../')
import passive_generator 
import optimization
from core import DEVICE
from alpha_batch import AlphaBatch

# load pre-trained reconstruction model in eval mode
model_path = '../models/Model2.pt'
model= torch.load(model_path)                  
model.eval()

# use generator to generate a batch of beam possitions
batch_size = 10000
generator = passive_generator.RandomBeamPositionGenerator()
alpha_batch = AlphaBatch(generator.generate_set(batch_size))

# fit
pressure = torch.tensor(random.uniform(10,50), dtype=torch.float32)
collimator_length = torch.tensor(random.uniform(5,50), dtype=torch.float32)

VolumeWrapper = optimization.AbsVolumeWrapper(pressure, collimator_length, alpha_batch, model,  lr=0.1, epochs=1000,  device=DEVICE)
loss_values, p_values, d_values = VolumeWrapper.fit()


plot = True
if plot:
    VolumeWrapper.plot_loss(loss_values)
    VolumeWrapper.plot_pressure(p_values)
    VolumeWrapper.plot_col_lenght(d_values)
