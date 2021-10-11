import torch.nn.modules as nn

from .magnified_loss import MagnifiedMSELoss
from .pose_loss import PoseLoss

from pytorch_helper.settings.spaces import Spaces


@Spaces.register(Spaces.NAME.LOSS_FN, 'BEVLoss')
class BEVLoss(nn.Module):
    def __init__(self, magnitude_scale, loss_weights):
        super(BEVLoss, self).__init__()
        self.magnitude_scale = magnitude_scale
        self.loss_weights = loss_weights

        self.map_loss_fn = MagnifiedMSELoss(magnitude_scale)
        self.cnt_mae_fn = nn.L1Loss()
        self.cnt_mse_fn = nn.MSELoss()
        self.pose_loss_fn = PoseLoss(loss_weights['pose-height'])

    def forward(self, pred_dict, target_dict):
        loss_dict = dict()
        for k in ['feet_map', 'head_map', 'bev_map']:
            if k in pred_dict and pred_dict[k] is not None:
                loss_dict[k] = self.map_loss_fn(pred_dict[k], target_dict[k])
                pred_cnt = self.get_cnt(pred_dict[k], self.magnitude_scale)
                target_cnt = self.get_cnt(target_dict[k], 1.)
                loss_dict[f'{k}-cnt-mae'] = self.cnt_mae_fn(
                    pred_cnt, target_cnt
                )
                loss_dict[f'{k}-cnt-mse'] = self.cnt_mse_fn(
                    pred_cnt, target_cnt
                )

        if 'camera_height' in pred_dict:
            loss_dict.update(self.pose_loss_fn.forward(
                pred_dict['camera_height'], pred_dict['camera_angle'],
                target_dict['camera_height'], target_dict['camera_angle']
            ))
        # sum of loss
        loss_all = 0
        for k, w in self.loss_weights.items():
            if k in loss_dict:
                loss_all += loss_dict[k] * w
        loss_dict['all'] = loss_all
        return loss_dict

    @staticmethod
    def get_cnt(arr, scale):
        bs = arr.size(0)
        return (arr.view(bs, -1) / scale).sum(-1)
