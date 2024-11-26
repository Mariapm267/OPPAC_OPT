from abc import ABCMeta, abstractmethod
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from volume import Volume
from typing import Callable, Dict, Optional, Union

"""
Provides loss functions for evaluating the performance of detector and inference configurations
"""

__all__ = ["AbsDetectorLoss", "BeamPositionLoss"]

class AbsDetectorLoss(nn.Module, metaclass=ABCMeta):
   def __init__(self,
                use_cost: Optional[bool] = False,
                #target_budget: Optional[float],
                #budget_smoothing: float = 10,
                #cost_coef: Optional[Union[Tensor, float]] = 1,
                debug = False):
      super().__init__()
      self.use_cost = use_cost
      #self.target_budget = target_budget
      #self.budget_smoothing: budget_smoothing
      #self.cost_coef = cost_coef
      #self.debug = debug

   @abstractmethod
   def _get_inference_loss(self, pred: Tensor, beam: Tensor) -> Tensor:
      r"""
      Inheriting class must override this, in order to actually compute the inference part of the loss function.
      The inference part is the only mandatory portion of the loss.
      """
      pass

   def forward(self, pred: Tensor, beam: Tensor) -> Tensor:
      r"""
      Computes the loss for the volume that is fed to the loss, using the current state of the detector.

      Arguments:
          pred: the predictions from the inference
          volume: Volume containing the passive volume that was being predicted and the detector being optimized
      """

      self.sub_losses = {}
      self.sub_losses["error"] = self._get_inference_loss(pred, beam)
      self.sub_losses["cost"] = self._get_cost_loss(beam) if self.use_cost else None
      return self.sub_losses["error"] + self.sub_losses["cost"] if self.use_cost else self.sub_losses["error"]

   
   def _get_budget_coef(self, cost: Tensor) -> Tensor:
      r"""
      Computes the budget loss term from the current cost of the detectors.
      Using the basic configuration from TomOpt paper, i.e. switch-on near target budget, plus linear/smooth increase above budget

      Arguments:
          cost: the current cost of the detector in currency units

      Returns:
          The budget loss component
      """

       
      if self.target_budget is None:
         return cost.new_zeros(1)

      if self.steep_budget:
         d = self.budget_smoothing * (cost - self.target_budget) / self.target_budget
         if d <= 0:
            return 2 * torch.sigmoid(d)
         else:
            return 1 + (d / 2)
      else:
         d = cost - self.target_budget
      return (2 * torch.sigmoid(self.budget_smoothing * d / self.target_budget)) + (F.relu(d) / self.target_budget)

   
   def _get_cost_loss(self, volume: Volume) -> Tensor:
      r"""
      Computes the budget term of the loss. For the moment, optimization will likely be run with use_cost=False,
      because the only parameter that acts on the cost is the pressure, and high pressure will be always favoured,
      resulting in a trivial answer. Down the line will explore if there are unexpected phenomena here.

      Arguments:
          volume: Volume containing the detectors that are being optimized

      Returns:
          The loss term for the cost of the detectors
      """
      
      cost = volume.get_cost()
      cost_loss = self._get_budget_coef(cost) * self.cost_coeff
      if self.debug:
         print(
            f'cost {cost}, cost coef {self.cost_coef}, budget coef {self._get_budget_coef(cost)}. error loss {self.sub_losses["error"]}, cost loss {cost_loss}'
         )

      return cost_loss

   
class BeamPositionLoss(AbsDetectorLoss):
   def _get_inference_loss(self, pred: Tensor, beam: Tensor) -> Tensor: # to be adapted
      return F.mse_loss(pred, beam, reduction="mean")

