class AbsDetectorLoss(nn.Module, metaclass=ABCMeta):
   def __init__(self):
     pass


class BasicLoss(AbsDetectorLoss):
   def _get_inference_loss(self, pred: Tensor, volume: Volume) -> Tensor: # to be adapted
     pass
