import numpy as np
import torch
import torch.nn.functional as F
from pytorch_helper.settings.spaces import Spaces
from pytorch_helper.task import Batch
from pytorch_helper.task import Task
from pytorch_helper.utils.log import get_logger

from settings.options import BEVNetTrainRoutine
from tasks.helper import assemble_3in1_image

__all__ = ['BEVNetTask']

logger = get_logger(__name__)


@Spaces.register(Spaces.NAME.TASK, ['BEVNetTask'])
class BEVNetTask(Task):

    def model_forward_backward(self, batch: Batch, backward=False):
        image = batch.gt['image']
        fu, fv = batch.gt['camera_fu'], batch.gt['camera_fv']
        h_cam = None
        p_cam = None
        if not self.current_train_routine.use_inferred_pose:
            h_cam = batch.gt['camera_height']
            p_cam = batch.gt['camera_angle']
        pred = self.model.forward(image, fu, fv, h_cam, p_cam)
        loss = self.loss_fn.forward(pred, batch.gt)
        if backward:
            if isinstance(loss, dict):
                loss['all'].backward()
            else:
                loss.backward()

        batch.pred = pred
        batch.loss = self.sync_value(loss)
        batch.size = image.size(0)
        return batch

    def post_processing(self, batch):
        for k in ['head_map', 'feet_map', 'bev_map']:
            if k in batch.pred and batch.pred[k] is not None:
                batch.pred[k] = F.relu(batch.pred[k])
                batch.pred[k] /= self.loss_fn.map_loss_fn.magnitude_scale
        return batch

    def test(self, batch: Batch) -> Batch:
        if self._option.train:
            return super(BEVNetTask, self).test(batch)
        with torch.no_grad():
            return self.post_processing(self.model_forward_backward(batch))

    def run_test(self):
        self.keep_model_output = self._option.test_option.save_model_output
        summary = super(BEVNetTask, self).run_test()
        if self.option.test_option.save_model_output:
            import os
            import h5py
            from pytorch_helper.utils.log import get_datetime

            path = os.path.join(self._option.output_path_test,
                                'model-output.h5')
            logger.info(f'Saving model output to {path}')
            h5file = h5py.File(path, 'w')
            for key, value in summary.items():
                if key.endswith('map-loss'):
                    value /= self.loss_fn.map_loss_fn.magnitude_scale ** 2
                h5file.attrs[key] = value
            h5file.attrs['summary-ordered-keys'] = list(summary.keys())
            h5file.attrs['datetime_test'] = get_datetime()
            for name, data in self.model_output_dict.items():
                logger.info(f'Saving {name}')
                h5file.create_dataset(
                    name, data.shape, data.dtype, data, compression='gzip'
                )
            h5file.close()
        return summary

    def collect_model_output(self, batch):
        super(BEVNetTask, self).collect_model_output(batch)
        for key in ['scene_id', 'image_id']:
            self.model_output_dict[key].append(batch.gt[key].cpu().numpy())

    def setup_before_stage(self):
        super(BEVNetTask, self).setup_before_stage()
        if self.cur_stage == self.STAGE.TRAIN:
            loss_weights = self.current_train_routine.loss_weight_changes
            if loss_weights is not None:
                self.loss_fn.loss_weights.update(loss_weights)
                logger.info(f'change loss weight: {loss_weights}')
        elif not self._option.train:  # Test Task
            self.current_train_routine = BEVNetTrainRoutine(epochs=self.epoch)
            if self._option.test_option.test_all:
                self.cur_dataloader = self.dataloader.all_loader

    def rank0_update_logging_in_stage(self, batch):
        super(BEVNetTask, self).rank0_update_logging_in_stage(batch)

        if self.in_stage_logged:
            return
        # log once per epoch
        self.in_stage_logged = True

        if self._option.train:
            index = np.random.randint(
                low=0, high=batch.gt['image'].size(0)
            )
            # 3in1 image: input - gt - pred
            batch.gt['image'] = self.dataloader.denormalize(batch.gt['image'])
            for tag, image in self.get_3in1_images(
                self.cur_stage.value, batch, index
            ).items():
                self.tboard.add_image(tag, image, self.epoch)
            # add head attention map
            key = 'height_attn'
            if key in batch.pred:
                heights = torch.tensor(self.unwrapped_model.head_heights)
                heights = heights[torch.argmax(batch.pred[key], dim=1)]
                self.tboard.add_histogram(
                    f'{self.cur_stage.value}/height-attention', heights,
                    self.epoch
                )

    def get_3in1_images(self, tag, result, index=0):
        gt_batch = {k: v.data.cpu() for k, v in result.gt.items()}
        pred_batch = {
            k: v.data.cpu() for k, v in result.pred.items()
            if v is not None
        }

        images = dict()
        for k, v in pred_batch.items():
            if v is None or k not in ['head_map', 'feet_map', 'bev_map']:
                continue
            gt_map = gt_batch[k][index]
            pred_map = v[index]
            loss = self.loss_fn.map_loss_fn(
                pred_batch[k][index], gt_batch[k][index]
            ).item()
            pd_height = pred_batch['camera_height'][index].item()
            pd_angle = pred_batch['camera_angle'][index].item() / np.pi * 180
            gt_height = gt_batch['camera_height'][index].item()
            gt_angle = gt_batch['camera_angle'][index].item() / np.pi * 180
            pose_loss = self.loss_fn.pose_loss_fn(
                pred_batch['camera_height'][[index]],
                pred_batch['camera_angle'][[index]],
                gt_batch['camera_height'][[index]],
                gt_batch['camera_angle'][[index]]
            )
            pose_height_loss = pose_loss['pose-height'].item()
            pose_angle_loss = pose_loss['pose-angle'].item()
            titles = [
                # input
                f'input, map loss={loss:.2e}\n'
                f'height loss={pose_height_loss:.2e}, '
                f'angle loss={pose_angle_loss:.2e}',
                # gt
                f'g.t. camera height={gt_height:.2e}m\n'
                f'camera angle={gt_angle:.2e}',
                # pred
                f'p.d. camera height={pd_height:.2e}m\n'
                f'camera angle={pd_angle:.2e}'
            ]
            images[f'{tag}/{k}'] = assemble_3in1_image(
                gt_batch['image'][index].cpu(), gt_map, pred_map, titles
            )
        return images
