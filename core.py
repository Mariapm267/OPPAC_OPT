import torch

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") # Check if TomOpt bug #53 affects us

