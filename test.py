r"""
Tests. Not formally unit tests, but at some point maybe they will become unit tests.
"""

import torch

from volume import Volume

print("Test: volume (no clamping needed)")
v = Volume(torch.tensor([0.3, 9.8]), pressure=torch.tensor([25.]), collimator_length=torch.tensor([0.3]))

tb = v.get_true_beam_position().detach()
v.clamp_parameters()
ctb = v.get_true_beam_position().detach()

assert(torch.all(tb == ctb))

print("PASSED")

print("Test: volume (clamping needed)")
v = Volume(torch.tensor([-256., 9.8]), pressure=torch.tensor([71.]), collimator_length=torch.tensor([134.]))

tb = v.get_true_beam_position().detach()
v.clamp_parameters()
ctb = v.get_true_beam_position().detach()

assert(torch.all(tb == ctb))

print("PASSED")
