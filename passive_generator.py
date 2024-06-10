from abc import ABCMeta, abstractmethod
from typing import Generator, List, Optional, Tuple, Union
from torch import Tensor
import torch

r"""
Provides classes that generate and yield passive volume layouts
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
    r"""
    Class that generates random beam positions.
    """
    def __init__(self):
        r"""
        Initializes the class, preparing it to generate a given number of volumes
        """
        super().__init__()

    def _generate(self) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
        Returns:
            generator: A function that provides an xy tensor of beam position, when called.
            Target: None
        """
        def generator() -> Tensor:
            xys = torch.rand(2) * 8000 - 4000
            return xys

        return generator, None

