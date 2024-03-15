r"""
Tests. Not formally unit tests, but at some point maybe they will become unit tests.
"""

import torch

from volume import Volume

v = Volume(torch.tensor([0.3, 9.8]), pressure=25, collimator_length=0.3)
