import os

import torch
import torch.nn.modules as nn
from scipy.spatial.distance import directed_hausdorff
from pytorch_helper.settings.spaces import Spaces

from .bev_graph import BEVGraph
from .nms_head import NMSHead


@Spaces.register(Spaces.NAME.METRIC, 'IndividualDistance')
class IndividualDistance(nn.Module):
    def __init__(self, dist_type, active_only, nms_head_kwargs):
        super(IndividualDistance, self).__init__()
        self.dist_type = dist_type
        self.active_only = active_only
        self.nms_head = NMSHead(**nms_head_kwargs)

    @staticmethod
    def hausdorff(u, v):
        if isinstance(u, torch.Tensor):
            u = u.cpu().numpy()
            v = v.cpu().numpy()
        return max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])

    @staticmethod
    def chamfer(u, v) -> torch.Tensor:
        if u.shape[0] == 0:
            u = torch.zeros(1, 2).to(u.device)
        if v.shape[0] == 0:
            v = torch.zeros(1, 2).to(v.device)
        pair_dist = torch.norm(u.unsqueeze(1) - v.unsqueeze(0), dim=2)
        chamfer_dist = pair_dist.min(0)[0].mean() + pair_dist.min(1)[0].mean()
        return chamfer_dist.item()

    def get_metric_output_path(self, output_path, mkdir=False):
        path = os.path.join(
            output_path, type(self).__name__, self.dist_type
        )
        if mkdir:
            os.makedirs(path, exist_ok=True)
        return path

    def forward(self, pred, gt):
        result = dict(
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
        # calculate distance
        result['summary'] = self.summarize(result)

        return result

    def summarize(self, coords, summary=None):
        if isinstance(coords, dict):
            pred_coord = coords['pred']['world_coords']
            target_coord = coords['gt']['world_coords']
            if summary is None:
                summary = dict(distance=[])
                if self.active_only:
                    summary['graph'] = dict(gt=[], pred=[])
            for pred, target in zip(pred_coord, target_coord):
                self.summarize((pred, target), summary)
            summary[f'{self.dist_type}-distance'] = summary.pop('distance')
            return summary

        pred_coord, target_coord = coords[0].T, coords[1].T  # (n, 2)
        if self.active_only:
            pred_graph = BEVGraph(pred_coord, **self.args)
            target_graph = BEVGraph(target_coord, **self.args)
            pred_coord = pred_coord[pred_graph.individuals]
            target_coord = target_coord[target_graph.individuals]
            summary['graph']['gt'].append(target_graph)
            summary['graph']['pred'].append(pred_graph)
        if self.dist_type == 'chamfer':
            summary['distance'].append(self.chamfer(pred_coord, target_coord))
        elif self.dist_type == 'hausdorff':
            summary['distance'].append(self.hausdorff(pred_coord, target_coord))
        else:
            raise ValueError(f"Unknown dist_type: {self.dist_type}")
        return summary
