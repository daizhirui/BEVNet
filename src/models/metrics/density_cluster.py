import os

import numpy as np
import torch.nn.modules as nn

from models.bevnet.bev_transform import BEVTransform
from .conv_threshold import ConvThreshold
from .kernels import get_grids
from .nms_head import NMSHead
from pytorch_helper.settings.spaces import Spaces


@Spaces.register(Spaces.NAME.METRIC, 'DensityCluster')
class DensityCluster(nn.Module):
    def __init__(
        self, risk_threshold, kernel_name, kernel_kwargs, nms_head_kwargs
    ):
        super(DensityCluster, self).__init__()
        self.conv = ConvThreshold(risk_threshold, kernel_name, kernel_kwargs)
        self.bev_transform = BEVTransform()
        self.loss_fn = nn.MSELoss()
        self.nms_head = NMSHead(**nms_head_kwargs)

    def get_metric_output_path(self, output_path, mkdir=True):
        path = os.path.join(
            output_path, type(self).__name__, self.conv.kernel_name
        )
        if mkdir:
            os.makedirs(path, exist_ok=True)
        return path

    def forward(self, pred, gt, dist_metric_result=None):
        if dist_metric_result is None:
            dist_metric_result = dict(
                pred=self.nms_head.forward(
                    input_map=pred['bev_map'],
                    bev_scale=pred['bev_scale'],
                    bev_center=pred['bev_center']
                ),
                gt=self.nms_head.forward(
                    input_map=gt['bev_map'],
                    bev_scale=gt['bev_scale'],
                    bev_center=gt['bev_center']
                )
            )
        # process the network output, generate a series maps, masks and etc.
        data = dict(pred=pred, gt=gt)
        metric_result = dict(
            pred=self.process(data, 'pred', dist_metric_result),
            gt=self.process(data, 'gt', dist_metric_result)
        )

        # get the metric summary and save visualization
        metric_result['summary'] = dict(
            loss=self.get_loss(metric_result),
            iou=self.get_iou(metric_result)
        )

        for j in ['gt', 'pred']:
            metric_result[j]['global_risk'] = \
                metric_result[j]['global_risk'].cpu().numpy()
        return metric_result

    def process(self, data, key, dist_metric_result=None) -> dict:
        bev_map = data[key]['bev_map']
        bev_scale = data[key]['bev_scale']
        h_cam = data[key]['camera_height']
        p_cam = data[key]['camera_angle']
        fu = data['gt']['camera_fu']
        fv = data['gt']['camera_fv']

        risk_map_bev, risk_mask_bev = self.conv(bev_map, bev_scale, True)

        bs = bev_map.size(0)
        global_risk = (bev_map * risk_mask_bev).view(bs, -1).sum(-1)
        area = bev_map.size(-1) * bev_map.size(-2) * bev_scale * bev_scale
        global_risk = global_risk / area  # num of violation per squared meter

        params = dict(
            plane_height=0, h_cam=h_cam, p_cam=p_cam, fu=fu, fv=fv, i2b=False
        )
        risk_map_iv = self.bev_transform(risk_map_bev, **params)[0]

        if dist_metric_result is None:
            individual_risks = [[]] * risk_mask_bev.size(0)
        else:
            bev_coords = dist_metric_result[key]['bev_coords']
            world_coords = dist_metric_result[key]['world_coords']
            feet_iv_coords = self.get_feet_iv_coords(data, key, world_coords)
            individual_risks = self.get_individual_risks(
                risk_map_bev, bev_map, bev_coords, feet_iv_coords
            )

        out = dict(
            risk_map_bev=risk_map_bev,
            risk_map_iv=risk_map_iv,
            risk_mask_bev=risk_mask_bev,
            global_risk=global_risk,
            area=area,
            individual_risks=individual_risks
        )

        return out

    def get_loss(self, metric_result):
        loss = dict()
        pred_dict, gt_dict = metric_result['pred'], metric_result['gt']
        k = 'global_risk'
        loss[k] = self.loss_fn(pred_dict[k], gt_dict[k])
        return loss

    @staticmethod
    def get_iou(metric_result):
        k = 'risk_mask_bev'
        bs = metric_result['pred'][k].size(0)
        y = metric_result['pred'][k].view(bs, -1)
        t = metric_result['gt'][k].view(bs, -1)
        inter = (y * t).gt(0).int().sum(dim=1)
        union = (y + t).gt(0).int().sum(dim=1)
        iou = inter.float() / union.float()
        iou[union == 0] = 1.0
        return iou.cpu().numpy()

    def get_feet_iv_coords(self, data, key, world_coords):
        homo_mat = self.bev_transform.get_bev_param(
            input_size=data[key]['bev_map'].size()[-2:],
            h_cam=data[key]['camera_height'],
            p_cam=data[key]['camera_angle'],
            fu=data['gt']['camera_fu'],
            fv=data['gt']['camera_fv'],
            plane_h=0,
            w2i=True
        )[0]
        result = [
            self.bev_transform.world_coord_to_image_coord(
                world_coords[i].unsqueeze(0), homo_mat[[i]]
            )[0, :2] for i in range(homo_mat.size(0))
        ]
        return result

    @staticmethod
    def get_individual_risks(risk_map, bev_map, bev_coords, feet_iv_coords):
        # kernel used for generating g.t. bev map
        r = 15
        k = 2 * r + 1
        # create the kernel
        param = (get_grids(size=k) - r).pow(2).sum(dim=0).le(r * r)
        param = param.float().view(1, 1, 31, 31)
        kernel = nn.Conv2d(1, 1, k, padding=r, bias=False, groups=1).eval()
        kernel.weight.data = param
        # calculate the risk
        kernel = kernel.to(risk_map.device)
        tmp = risk_map * bev_map
        indiv_risk_map = kernel(tmp)

        individual_risks = []
        for i in range(len(bev_coords)):
            feet_iv_coord = feet_iv_coords[i][:2].cpu().numpy()
            u = bev_coords[i][0, :]
            v = bev_coords[i][1, :]
            indiv_risks = indiv_risk_map[i, 0, v, u].cpu().numpy()
            indiv_risks = np.concatenate([
                feet_iv_coord.T, indiv_risks[:, np.newaxis]
            ], axis=1)
            individual_risks.append(indiv_risks.tolist())
        # individual_risks = [[
        #     (
        #         feet_iv_coords[i][0, k].item(),
        #         feet_iv_coords[i][1, k].item(),
        #         indiv_risk_map[
        #             i, 0, bev_coords[i][1, k], bev_coords[i][0, k]
        #         ].item()
        #     )
        #     for k in range(bev_coords[i].size(1))
        # ] for i in range(len(bev_coords))]
        return individual_risks
