from torch.nn.functional import interpolate

from pytorch_helper.settings.spaces import Spaces
from tasks.ivnet import IVNetTask

__all__ = ['CSRNetTask']


@Spaces.register(Spaces.NAME.TASK, 'CSRNetTask')
class CSRNetTask(IVNetTask):

    def model_forward_backward(self, batch, backward=False):
        image = batch.gt['image']

        if self._option.train:
            iv_map = interpolate(batch.gt['head_map'], size=(48, 64),
                                 mode='bilinear', align_corners=True)
            pred_iv_map = self.model.forward(image, upsample=False)
        else:
            iv_map = batch.gt['head_map']
            pred_iv_map = self.model(image)

        loss = self.loss_fn(pred_iv_map, iv_map)
        if backward:
            loss.backward()

        if self._option.train:
            pred_iv_map = interpolate(pred_iv_map, size=(384, 512),
                                      mode='bilinear', align_corners=True)
        batch.pred = pred_iv_map
        batch.loss = self.sync_value(loss)
        batch.size = image.size(0)
        return batch
