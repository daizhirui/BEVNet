from pytorch_helper.settings.options import TaskMode
from pytorch_helper.settings.spaces import Spaces
from pytorch_helper.task import Batch
from tasks.bevnet import BEVNetTask


@Spaces.register(Spaces.NAME.TASK, 'IV2BEVTask')
class IV2BEVTask(BEVNetTask):

    def __init__(self, task_option):
        task_option.task_mode = TaskMode.TEST
        super(IV2BEVTask, self).__init__(task_option)
        self.cc_oracle = task_option.test_option.cc_oracle
        self.pose_oracle = task_option.test_option.pose_oracle
        self.iv_map = task_option.test_option.iv_map
        self.plane_height = 1.75
        if self.iv_map == 'feet_map':
            self.plane_height = 0.
        self.map_scale = self.loss_fn.map_loss_fn.magnitude_scale

    def model_forward_backward(
        self, batch: Batch, backward=False
    ):
        image = batch.gt['image']
        fu, fv = batch.gt['camera_fu'], batch.gt['camera_fv']
        pred = self.model.forward(image, fu, fv)

        if self.pose_oracle:
            h_cam = batch.gt['camera_height']
            p_cam = batch.gt['camera_angle']
        else:
            h_cam = pred['camera_height']
            p_cam = pred['camera_angle']

        if self.cc_oracle:
            input_map = batch.gt[self.iv_map] * self.map_scale
        else:
            input_map = pred[self.iv_map]

        bevmap, bev_scale, bev_center = self.unwrapped_model.bev_transform(
            input_map, self.plane_height, h_cam, p_cam, fu, fv
        )
        roi = self.unwrapped_model.bev_transform.get_iv_roi(
            bevmap.size(), h_cam, p_cam, fu, fv, filled=True
        )
        bs = input_map.size(0)
        s_iv = (input_map * roi.float()).view(bs, -1).sum(-1)
        s_bev = bevmap.view(bs, -1).sum(-1) + 1e-16
        bevmap *= (s_iv / s_bev).view(bs, 1, 1, 1)

        pred = dict(
            bev_map=bevmap,
            camera_height=h_cam,
            camera_angle=p_cam,
            bev_scale=bev_scale,
            bev_center=bev_center
        )
        pred[self.iv_map] = input_map
        loss = self.sync_value(self.loss_fn.forward(pred, batch.gt))

        batch.pred = pred
        batch.loss = self.sync_value(loss)
        batch.size = image.size(0)
        return batch
