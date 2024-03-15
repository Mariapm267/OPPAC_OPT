from typing import Any, List, Optional, Tuple

import torch
from torch import nn
from alpha_batch import AlphaBatch



r"""
Provides implementation of wrapper classes for the volume made of passive volume and optimizable detector.
"""

__all__ = ["Volume"]

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") # Check if TomOpt bug #53 affects us

class Volume(nn.Module):
  def __init__(self, beam_xy: Tensor, pressure: float, collimator_length: Union[Tensor, float], budget: Optional[float] = None, device: torch.device = DEVICE):
  r"""
   Initializes the volume with a certain beam position, pressure, and collimators length

    Arguments:
        beam_position: the tensor of the (x,y) position of the beam
        pressure: pressure of the gas in the volume
        collimator_length: length of the collimators. Can be a float (same collimator length on all four sides) or a tensor of size four (each side has one collimator length)
        budget: optional budget of the detector in currency units.
            Supplying a value for the optional budget, here, will prepare the volume to learn budget assignments to the detectors,
            and configure the detectors for the budget.
    """

    super().__init__()
    self.beam_xy = beam_xy
    self._device = device
    self.pressure = pressure
    self.collimator_length = collimator_length
    self.budget = None if budget is None else torch.tensor(budget, device=self.device)
  
  def forward(self, alpha: AlphaBatch) -> None:
    r"""
    Propagates the alphas to generate the photomultipliers readouts
    
    Arguments:
        alpha: the incoming batch of alphas
    """
    pass

  def draw(self) -> None:
    r"""
        Draws the layers/panels pertaining to the volume.
        When using this in a jupyter notebook, use "%matplotlib notebook" to have an interactive plot that you can rotate.
    """
    pass

  def get_true_beam_position(self) -> Tensor:
    r"""
    Gives the true beam position, to be used for the inference part of the loss function

    Arguments:
        None

    Returns:
        Tensor of (x,y) position of the beam
    """
    return self.beam_xy
  
  def get_cost(self):
    pass
