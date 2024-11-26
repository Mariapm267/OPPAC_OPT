from core import DEVICE
import torch
from torch import Tensor
import passive_generator 


class AlphaBatch:
    def __init__(self, xy_e: Tensor, device: torch.device = torch.device('cpu')):
        self.device = device
        self._alphas = xy_e.to(self.device)
        self.beam_xy = xy_e

        def __repr__(self) -> str:
            return f"Batch of {len(self)} alphas"

        def __len__(self) -> int:
            return len(self._alphas)
    
    def get_beam_xy(self) -> Tensor:
        return self.beam_xy
      
    def get_photon_distribution(self) -> Tensor:
      '''Not implemented for now, as the reconstruction process is made separately'''
      pass
    
    