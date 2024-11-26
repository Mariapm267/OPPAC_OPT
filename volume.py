from typing import Any, Union, List, Optional, Tuple

from core import DEVICE
import torch
from torch import nn, Tensor
from alpha_batch import AlphaBatch


r"""
Provides implementation of wrapper classes for the volume made of passive volume and optimizable detector.
"""

__all__ = ["Volume"]

class Volume(nn.Module):
    def __init__(self, pressure: Tensor, collimator_length: Tensor, alpha_batch: Tensor, budget: Optional[float] = None, pressure_range: Tensor = Tensor([10., 50.]), collimator_length_range: Tensor = Tensor([5., 50.]), device: torch.device = DEVICE):
        r"""
        Initializes the volume with a certain beam position, pressure, and collimators length
    
        Arguments:
            pressure: pressure of the gas in the volume
            collimator_length: length of the collimators. Tensor of size 1.
            budget: optional budget of the detector in currency units.
                Supplying a value for the optional budget, here, will prepare the volume to learn budget assignments to the detectors,
                and configure the detectors for the budget.

        NOTE: PROBABLY REFACTOR (passive/active, to clamp correctly and have multiple passives for one active)
        """
    
        super().__init__()
        self.alpha_batch = alpha_batch
        self._device = device
        self.pressure = pressure
        self.collimator_length = collimator_length
        self.budget = None if budget is None else torch.tensor(budget, device=self.device)
        self.pressure_range = pressure_range
        self.collimator_length_range = collimator_length_range
  
    def forward(self, alpha: AlphaBatch) -> None:
        r"""
        Propagates the alphas to generate the photomultipliers readouts
        
        Arguments:
            alpha: the incoming batch of alphas
        """
        pass
  

    def clamp_parameters(self) -> None:
        r"""
        Checks that the parameters are within the specifications of the volume,
        and clamps them when necessary.
        ATTENTION: THIS MUST BE EXPLICITLY CALLED IN THE OPTIMIZATION LOOP,
        AFTER EACH PARAMETERS UPDATE
        """
        with torch.no_grad():
            self.pressure.clamp_(min=self.pressure_range[0], max=self.pressure_range[1])
            self.collimator_length.clamp_(min=self.collimator_length_range[0], max=self.collimator_length_range[1])
    
    def get_true_beam_position(self) -> Tensor:
        r"""
        Gives the true beam position, to be used for the inference part of the loss function
        
        Arguments:
            None
        
        Returns:
            Tensor of (x,y) position of the beam
        """
        return self.beam_xy

    def get_pressure(self) -> Tensor:
        r"""
        Returns the current pressure
        """
        return self.pressure

    def get_collimator_length(self) -> Tensor:
        r"""
        Returns the current collimator length
        """
        return self.collimator_length
    

