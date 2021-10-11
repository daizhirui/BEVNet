import torch.nn.modules as nn
from pytorch_helper.settings.spaces import Spaces


@Spaces.register(Spaces.NAME.LOSS_FN, 'MagnifiedMSELoss')
class MagnifiedMSELoss(nn.Module):
    def __init__(self, magnitude_scale):
        super(MagnifiedMSELoss, self).__init__()
        self.loss_fn = nn.MSELoss()
        self.magnitude_scale = magnitude_scale

    def forward(self, pred, target):
        return self.loss_fn(pred, target * self.magnitude_scale)
