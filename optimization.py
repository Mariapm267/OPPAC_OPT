import torch


__all__ = ["AbsVolumeWrapper"]


r"""
Provides wrapper classes for optimising detectors and other quality-of-life methods

FitParams and AbsVolumeWrapper are modified versions of the FitParams in LUMIN (https://github.com/GilesStrong/lumin/blob/v0.7.2/lumin/nn/models/abs_model.py#L16)
and Model in LUMIN (https://github.com/GilesStrong/lumin/blob/master/lumin/nn/models/model.py#L32), distributed under the following licence:
    Copyright 2018 onwards Giles Strong

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

Usage is compatible with the AGPL licence under which OPPAC_opt is distributed.
Stated changes: adaption of FitParams to pass type-checking, heavy adaptation of Model to be suitable for task specific training
"""



    
class AbsVolumeWrapper(Volume):
    def __init__(self, initial_pressure, initial_collimator_length, alpha_batch, model,  lr=0.01, epochs=10000, device=DEVICE):
        super().__init__(initial_pressure, initial_collimator_length, alpha_batch, device=device)
        self.model = model
        self.alpha_batch = alpha_batch.to(device)
        self.pressure = initial_pressure.clone().requires_grad_(True).to(device)
        self.collimator_length = initial_collimator_length.clone().requires_grad_(True).to(device)
        self.trainable_inputs = torch.stack([self.collimator_length, self.pressure])
        self.optimizer = optim.Adam([self.collimator_length, self.pressure], lr=lr)
        self.loss_fn = nn.MSELoss()
        self.epochs = epochs

    def fit(self):
        loss_values = []
        p_values = []
        d_values = []
        for epoch in tqdm(range(self.epochs), desc="Searching for best parameters...", unit="epoch"):
            self.optimizer.zero_grad()
            
            expanded_trainable_inputs = torch.stack([self.pressure, self.collimator_length]).expand(len(self.alpha_batch), -1)
            inputs = torch.cat((expanded_trainable_inputs, self.alpha_batch), dim=1)
            
            predictions = self.model(inputs)
            loss = self.loss_fn(predictions, self.alpha_batch[:, -2:])
            
            loss.backward()
            self.optimizer.step()
            
            self.clamp_parameters()
            loss_values.append(loss.item())
            p_values.append(self.pressure.detach().cpu().item())
            d_values.append(self.collimator_length.detach().cpu().item())

        print(f'Best parameters are...\nPressure: {self.pressure.detach()} Torr\nCollimator length: {self.collimator_length.detach()} mm')
        return loss_values, p_values, d_values

    def plot_loss(self, loss_values, filename = 'loss_curve_OPT'):
        plt.figure()
        plt.plot(range(self.epochs), np.sqrt(loss_values) / 1000, label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss [cm]')
        plt.title('Optimisation Loss Function')
        plt.legend()
        plt.savefig(f'../../figs/{filename}.pdf', dpi=400)
        
    def plot_pressure(self, p_values, filename = 'pressure_curve_OPT'):
        plt.figure()
        plt.plot(range(self.epochs), p_values)
        plt.xlabel('Epoch')
        plt.ylabel('p [Torr]')
        plt.title('Evolution of pressure')
        plt.savefig(f'../../figs/{filename}.pdf', dpi=400)
        
    def plot_col_lenght(self, d_values, filename = 'd_curve_OPT'):
        plt.figure()
        plt.plot(range(self.epochs), d_values)
        plt.xlabel('Epoch')
        plt.ylabel('d [mm]')
        plt.title('Evolution of collimator length')
        plt.savefig(f'../../figs/{filename}.pdf', dpi=400)
        
    




# Lots of work to be done here
