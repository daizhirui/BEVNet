import torch.nn.modules as nn
from pytorch_helper.settings.spaces import Spaces


@Spaces.register(Spaces.NAME.LOSS_FN, 'PoseLoss')
class PoseLoss(nn.Module):
    def __init__(self, height_loss_weight):
        super(PoseLoss, self).__init__()
        self.loss_fn = nn.MSELoss()
        self.height_loss_weight = height_loss_weight

    def forward(self, pred_height, pred_angle, target_height, target_angle):
        loss_height = self.loss_fn(pred_height, target_height)
        loss_angle = self.loss_fn(pred_angle, target_angle)
        loss = loss_angle + loss_height * self.height_loss_weight
        return {
            'pose': loss,
            'pose-height': loss_height,
            'pose-angle': loss_angle
        }
