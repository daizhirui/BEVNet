import numpy as np
import torch

from pytorch_helper.settings.options import TaskOption
from pytorch_helper.settings.spaces import Spaces
from pytorch_helper.task import Task
from tasks.helper import assemble_3in1_image


@Spaces.register(Spaces.NAME.TASK, 'IVNetTask')
class IVNetTask(Task):

    def __init__(self, task_option: TaskOption):
        super(IVNetTask, self).__init__(task_option)
        self.iv_map_key = getattr(self.unwrapped_model, 'iv_map', None)

    def model_forward_backward(
        self, batch, backward=False
    ):
        image = batch.gt['image']
        iv_map = batch.gt[self.iv_map_key]

        pred_iv_map = self.model(image)
        loss = self.loss_fn(pred_iv_map, iv_map)
        if backward:
            loss.backward()

        batch.pred = pred_iv_map
        batch.loss = self.sync_value(loss)
        batch.size = image.size(0)
        return batch

    def get_3in1_images(self, tag, result, index=0):
        titles = ['input']
        batch = result.gt
        if 'head_map' in batch:
            titles.append('g.t. of head map')
            sub_tag = 'head_map'
            gt_map = batch['head_map'][[index]]
        elif 'feet_map' in batch:
            titles.append('g.t. of feet map')
            sub_tag = 'feet_map'
            gt_map = batch['feet_map'][[index]]
        else:
            raise ValueError('No head_map or feet_map in the mini-batch')
        pred_map = result.pred[[index]]

        with torch.no_grad():
            loss = self.loss_fn(pred_map, gt_map).cpu().item()
        titles.append(f'p.d. loss={loss:.2e}')

        return {
            f'{tag}/{sub_tag}': assemble_3in1_image(
                self.dataloader.denormalize(batch['image'][index]).cpu(),
                gt_map[0].cpu(), pred_map[0].cpu(), titles
            )
        }

    def rank0_update_logging_in_stage(self, result):
        super(IVNetTask, self).rank0_update_logging_in_stage(result)
        if self.in_stage_logged:
            return
        self.in_stage_logged = True

        if self._option.train:
            index = np.random.randint(
                low=0, high=result.gt['image'].size(0)
            )
            # 3in1 image: input - gt - pred
            for tag, image in self.get_3in1_images(
                self.cur_stage.value, result, index
            ).items():
                self.tboard.add_image(tag, image, self.epoch)
