import torch
import sys

# From repo
sys.path.append('../')
import passive_generator 
import optimization
from core import DEVICE

# load pre-trained reconstruction model in eval mode

model_path = '../models/Model2.pt'
model= torch.load(model_path)                  
model.eval()

# use generator to generate a batch of beam possitions
batch_size = 100000
generator = passive_generator.RandomBeamPositionGenerator()
alpha_batch = generator.generate_set(batch_size) 

# set initial values of pressure and collimator length
initial_pressure = torch.tensor(30.0, dtype=torch.float32)
initial_collimator_length = torch.tensor(10.0, dtype=torch.float32)


VolumeWrapper = optimization.VolumeOptimizer(initial_pressure, initial_collimator_length, alpha_batch, model,  lr=0.1, epochs=1000, device=DEVICE)
loss_values, p_values, d_values = VolumeWrapper.fit()

plot = True
if plot:
    VolumeWrapper.plot_loss(loss_values)
    VolumeWrapper.plot_pressure(p_values)
    VolumeWrapper.plot_col_lenght(d_values)
