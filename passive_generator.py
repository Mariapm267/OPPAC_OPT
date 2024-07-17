from abc import ABCMeta, abstractmethod
from typing import  Optional, Tuple
from torch import Tensor
import torch

r"""
Provides classes that generate the "pasive volume', which is the beam position in this case
"""

__all__ = ["AbsPassiveGenerator", "RandomBeamPositionGenerator"]
class AbsPassiveGenerator(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def _generate(self) -> Tuple[callable, Optional[Tensor]]:
        pass

    def generate(self) -> Tensor:
        f, _ = self._generate()
        return f()

    def get_data(self) -> Tuple[callable, Optional[Tensor]]:
        return self._generate()


class RandomBeamPositionGenerator(AbsPassiveGenerator):
    def __init__(self):
        super().__init__()

    def _generate(self) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
        Returns:
            generator: A function that provides an xy tensor of beam position, when called.
        """
        def generator() -> Tensor:
            xys = torch.rand(2) * 8000 - 4000     # random numbers from -4000 to 4000 
            return xys
        
        return generator, None
    
    def generate_set(self, n: int) -> Tensor:
        r"""
        Generates a set of n random beam positions.

        Args:
            n (int): Number of positions to generate.

        Returns:
            Tensor: A tensor of shape (n, 2) with random beam positions.
        """
        positions = [self.generate() for _ in range(n)]
        return torch.stack(positions)
