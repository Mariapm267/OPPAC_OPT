from abc import ABCMeta, abstractmethod
from typing import Generator, List, Optional, Tuple, Union
from torch import Tensor

r"""
Provides classes that generate and yield passive volume layouts
"""

__all__ = ["AbsPassiveGenerator", "RandomBeamPositionGenerator"]

class AbsPassiveGenerator(metaclass=ABCMeta):
    r"""
    Abstract base class for classes that generate new passive layouts.

    The :meth:`AbsPassiveGenerator._generate` method should be overridden to return:
    - A function that provides an xy tensor for the position of the beam
    - An optional "target" value for the layout
    """


    def __init__(self):
        pass

    @abstractmethod
    def _generate(self, Tuple[Tensor, Optional[Tensor]]):
        pass
        

    def generate(self) -> Tensor:
        r"""
        Returns:
            The layout function and no target
        """

        f, _ = self._generate()
        return f

    def get_data(self) -> Tensor:
        r"""
        Returns:
            RadLengthFunc: A function that provides an xy tensor for a given layer when called with its z position, length and width, and size.
            Target: An optional "target" value for the layout
        """
        return self._generate()
    
class RandomBeamPositionGenerator(AbsPassiveGenerator):
    r"""
    Class that generates random beam positions ("Volumes").
    """
    def __init__(self):
        r"""
        Initializes the class, preparing it to generate a given number of volumes

        """
        super().__init()__

    def _generate(self, size: Tuple[float, float] = Tuple(10.,10.)) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
        Generates new passive layouts where ever voxel is of a random material.

        Returns:
            generator: A function that provides an xy tensor of beam position, when called.
            Target: None
        """

        def generator(*, xy: Tensor) -> Tensor:
            xys = torch.rand(2) 
            return xys

        return generator, None

class BeamPositionYielder:
    r"""
    Dataset class that can either:
        Yield from a set of pre-specified passive-volume layouts, and optional targets
        Generate and yield random layouts and optional targets from a provided generator

    Arguments:
        passives: Either a list of passive-volume functions (and optional targets together in a tuple), or a passive-volume generator
        n_passives: if a generator is used, this determines the number of volumes to generator per epoch in training, or in total when predicting
        shuffle: If a list of pre-specified layouts is provided, their order will be shuffled if this is True
    """

    def __init__(
        self,
        passives: Union[List[Union[Tuple[Tensor, Optional[Tensor]], Tensor]], AbsPassiveGenerator, RandomBeamPositionGenerator],
        n_passives: Optional[int] = None,
        shuffle: bool = True,
    ):
        self.passives, self.n_passives, self.shuffle = passives, n_passives, shuffle
        if isinstance(self.passives, AbsPassiveGenerator) or isinstance(self.passives, RandomBeamPositionGenerator):
            if self.n_passives is None:
                raise ValueError("If a AbsPassiveGenerator class is used, n_passives must be specified")
            else:
                self.n_passives = len(self.passives)

    def __len__(self) -> int:
        return self.n_passives

    def __iter__(self) -> Generator[Tuple[RadLengthFunc, Optional[Tensor]], None, None]:
        if isinstance(self.passives, AbsPassiveGenerator) or isinstance(self.passives, RandomBeamPositionGenerator:
            for _ in range(self.n_passives):
                yield self.passives.get_data()
        else:
            if self.shuffle:
                shuffle(self.passives)
            for p in self.passives:
                if isinstance(p, tuple):
                    yield p
                else:
                    yield p, None
