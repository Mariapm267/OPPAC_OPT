from core import DEVICE
import torch
from torch import Tensor
import passive_generator 


class AlphaBatch:
    def __init__(self, batch_size: int, generator = passive_generator.RandomBeamPositionGenerator() ,device: torch.device = DEVICE):
        self.device = device
        self.batch_size = batch_size
        self.beam_xy = generator.generate_set(batch_size)
        self._alphas = self.beam_xy.to(device)
        print(f'Using {self.device}')
        
        def __repr__(self) -> str:
            return f"Batch of {len(self)} alphas"

        def __len__(self) -> int:
            return len(self._alphas)
    
    def get_beam_xy(self) -> Tensor:
        return self.beam_xy
      
    def get_photon_distribution(self) -> Tensor:
      '''Not implemented for now, as the reconstruction process is made separately'''
      return NotImplementedError
