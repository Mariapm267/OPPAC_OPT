r"""
Tests. Not formally unit tests, but at some point maybe they will become unit tests.
"""

import torch

from volume import Volume

print("Test: volume (no clamping needed)")
v = Volume(torch.tensor([0.3, 9.8]), pressure=torch.tensor([25.]), collimator_length=torch.tensor([0.3]))

tb = v.get_true_beam_position().detach()
p = v.get_pressure().detach()
cl = v.get_collimator_length().detach()
v.clamp_parameters()
ctb = v.get_true_beam_position().detach()
cp = v.get_pressure().detach()
ccl = v.get_collimator_length().detach()

assert(torch.all(tb == ctb) and torch.all(p==cp) and torch.all(cl == ccl))

print("\tPASSED")

print("Test: volume (clamping needed)")
v = Volume(torch.tensor([-256., 0.3]), pressure=torch.tensor([71.]), collimator_length=torch.tensor([134.]))

tb = v.get_true_beam_position().detach()
p = v.get_pressure().detach()
cl = v.get_collimator_length().detach()
v.clamp_parameters()
ctb = v.get_true_beam_position().detach()
cp = v.get_pressure().detach()
ccl = v.get_collimator_length().detach()

assert(torch.all(ctb == torch.tensor([0., 0.3])) and torch.all(cp==torch.tensor([50])) and torch.all(ccl == torch.tensor([50.])))

print("\tPASSED")



