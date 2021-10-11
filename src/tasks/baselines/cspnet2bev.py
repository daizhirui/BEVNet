import os
from collections import OrderedDict

import h5py
import numpy as np
import torch

from datasets.cityuhk.kernels import GaussianKernel
from models import BEVTransform
from models.cspnet.functions import parse_det_offset
from pytorch_helper.settings.spaces import Spaces
from pytorch_helper.task import Batch
from pytorch_helper.task import Task
from pytorch_helper.utils.io import save_dict_as_csv
from pytorch_helper.utils.log import get_datetime
from pytorch_helper.utils.log import get_logger
from pytorch_helper.utils.log import pbar
from pytorch_helper.utils.meter import Meter

__all__ = ['CSPNet2BEVTask']

logger = get_logger(__name__)


@Spaces.register(Spaces.NAME.TASK, 'CSPNet2BEVTask')
class CSPNet2BEVTask(Task):

    def __init__(self, task_option):
        assert task_option.test_option.iv_map == 'head_map', \
            'CSPNet only outputs IV map of heads'
        super(CSPNet2BEVTask, self).__init__(task_option)

        torch.cuda.set_device(self.option.cuda_ids[0])

        self.loss_fn = self.option.loss.build()
        if not self.option.test_option.pose_oracle:
            self.pose_net = self.option.pose_net.build()[0]
            self.pose_net.cuda()
            self.pose_net.eval()
        self.bev_transform = BEVTransform()

        gaussian_sigma = 5
        gaussian_kernel_size = 6 * gaussian_sigma + 1
        self.gaussian_conv = GaussianKernel(gaussian_kernel_size,
                                            gaussian_sigma, channels=1).cuda()

        self.option.dataloader.kwargs['batch_size'] = 1
        self.dataloader = self.option.dataloader.build()
        self.test_loader = self.dataloader.test_loader

        self.meter = Meter()
        self.in_stage_meter_keys = set()
        self.model_output_dict = dict()

    def run(self):
        self.model.eval()
        self.pose_net.eval()
        for batch in pbar(self.test_loader, desc='Test'):
            with torch.no_grad():
                result = self.model_forward_backward(Batch(batch))
            self.update_logging_in_stage(result)

        summary = self.summarize_logging_after_stage()
        path = os.path.join(self.option.output_path_test, 'test-summary.csv')
        save_dict_as_csv(path, summary)

        if self.option.test_option.save_model_output:
            for key, value in self.model_output_dict.items():
                self.model_output_dict[key] = np.concatenate(value, axis=0)
            path = os.path.join(self.option.output_path_test, 'model-output.h5')
            logger.info(f'Saving model output to {path}')
            h5file = h5py.File(path, 'w')
            for key, value in summary.items():
                h5file.attrs[key] = value
            h5file.attrs['summary-ordered-keys'] = list(summary.keys())
            h5file.attrs['datetime_test'] = get_datetime()
            for name, data in self.model_output_dict.items():
                logger.info(f'Saving {name}')
                h5file.create_dataset(
                    name, data.shape, data.dtype, data, compression='gzip'
                )
            h5file.close()

    def model_forward_backward(self, batch, backward=False):
        for k, v in batch.gt.items():
            batch.gt[k] = v.cuda()

        image = batch.gt['image']
        fu, fv = batch.gt['camera_fu'], batch.gt['camera_fv']
        bs, _, h, w = image.size()

        pos, height, offset = self.model(image)
        pred_camera_pose = self.pose_net(image)
        pred_h = pred_camera_pose['camera_height']
        pred_a = pred_camera_pose['camera_angle']
        homo_inv_mats, scales, centers = self.bev_transform.get_bev_param(
            [h, w], pred_h, pred_a, fu, fv, w2i=False
        )

        bboxes = parse_det_offset(
            pos.cpu().numpy(), height.cpu().numpy(), offset.cpu().numpy(),
            size=(h, w), score=0.1, down=4, nms_thresh=0.5
        )
        n_coords = len(bboxes)

        # batch size is 1 !!
        device = image.device
        pred_head_map = torch.zeros((1, 1, h, w)).to(device)
        pred_feet_map = torch.zeros((1, 1, h, w)).to(device)
        pred_bev_map = torch.zeros((1, 1, h, w)).to(device)

        if n_coords > 0:
            bboxes = torch.tensor(bboxes)

            u = ((bboxes[:, 0] + bboxes[:, 2]) / 2).long()
            v = bboxes[:, 1].long()
            roi = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            u = u[roi]
            v = v[roi]
            pred_head_map[0, 0, v, u] = 1

            feet_pixels = torch.zeros(1, 3, n_coords)
            feet_pixels[:, 2] = 1
            feet_pixels[0, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2
            feet_pixels[0, 1] = bboxes[:, 3]
            u = feet_pixels[0, 0].long()
            v = feet_pixels[0, 1].long()
            roi = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            u = u[roi]
            v = v[roi]
            pred_feet_map[0, 0, v, u] = 1

            world_coords_homo = self.bev_transform.image_coord_to_world_coord(
                feet_pixels.to(device), homo_inv_mats
            )
            pred_bev_coords = self.bev_transform.world_coord_to_bev_coord(
                (h, w), world_coords_homo, scales, centers
            )

            bev_coords = pred_bev_coords[0, :2].long()
            u = bev_coords[0]
            v = bev_coords[1]
            roi = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            u = u[roi]
            v = v[roi]
            pred_bev_map[0, 0, v, u] = 1

        pred_head_map = self.gaussian_conv(pred_head_map)
        pred_feet_map = self.gaussian_conv(pred_feet_map)
        pred_bev_map = self.gaussian_conv(pred_bev_map)

        batch.size = bs
        batch.pred = dict()
        batch.pred['camera_height'] = pred_h
        batch.pred['camera_angle'] = pred_a
        batch.pred['head_map'] = pred_head_map
        batch.pred['feet_map'] = pred_feet_map
        batch.pred['bev_map'] = pred_bev_map
        batch.pred['bev_scale'] = scales
        batch.pred['bev_center'] = centers

        # loss
        batch.loss = self.loss_fn(batch.pred, batch.gt)

        return batch

    def update_logging_in_stage(self, result):
        loss = result.loss
        for k, v in loss.items():
            if v is not None:
                key = f'test/{k}-loss'
                self.meter.record(
                    tag=key, value=v.item(), weight=result.size,
                    record_op=Meter.RecordOp.APPEND,
                    reduce_op=Meter.ReduceOp.SUM
                )
                self.in_stage_meter_keys.add(key)

        if self.option.test_option.save_model_output:
            for key in ['scene_id', 'image_id']:
                self.model_output_dict.setdefault(key, list()).append(
                    result.gt[key].cpu().numpy()
                )
            for key, data in result.pred.items():
                if isinstance(data, torch.Tensor):
                    self.model_output_dict.setdefault(key, list()).append(
                        data.cpu().numpy()
                    )

    def summarize_logging_after_stage(self):
        summary = OrderedDict()
        summary['name'] = self.option.name
        summary['datetime'] = self.option.datetime
        summary['epoch'] = 'NA'
        summary['pth_file'] = 'NA'
        for key in sorted(list(self.in_stage_meter_keys)):
            summary[key] = self.meter.mean(key)
        return summary

    def backup(self, immediate=False, resumable=True):
        pass
