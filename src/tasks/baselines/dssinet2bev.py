from pytorch_helper.task import Batch

from models import BEVTransform
from tasks.baselines.iv2bev import IV2BEVTask
from pytorch_helper.settings.spaces import Spaces
__all__ = ['DSSINet2BEVTask']


@Spaces.register(Spaces.NAME.TASK, 'DSSINet2BEVTask')
class DSSINet2BEVTask(IV2BEVTask):

    def __init__(self, task_option):
        super(DSSINet2BEVTask, self).__init__(task_option)
        if not self.option.test_option.pose_oracle:
            self.pose_net = self.option.pose_net.build()[0]
            self.pose_net.cuda()
            self.pose_net.eval()
        self.bev_transform = BEVTransform()

    def post_init(self, state_dict):
        assert self.option.test_option.iv_map == 'head_map', \
            'DSSINet only outputs IV map of heads'
        super(DSSINet2BEVTask, self).post_init(state_dict)

    def model_forward_backward(
        self, batch: Batch, backward=False
    ):
        image = batch.gt['image']
        fu, fv = batch.gt['camera_fu'], batch.gt['camera_fv']
        input_map = self.model.forward(image)

        if self.pose_oracle:
            h_cam = batch.gt['camera_height']
            p_cam = batch.gt['camera_angle']
        else:
            pred_pose = self.pose_net(image)
            h_cam = pred_pose['camera_height']
            p_cam = pred_pose['camera_angle']

        bevmap, bev_scale, bev_center = self.bev_transform(
            input_map, self.plane_height, h_cam, p_cam, fu, fv
        )
        roi = self.bev_transform.get_iv_roi(
            bevmap.size(), h_cam, p_cam, fu, fv, filled=True
        )
        bs = input_map.size(0)
        s_iv = (input_map * roi.float()).view(bs, -1).sum(-1)
        s_bev = bevmap.view(bs, -1).sum(-1) + 1e-16
        bevmap *= (s_iv / s_bev).view(bs, 1, 1, 1)

        pred = dict(
            head_map=input_map,
            bev_map=bevmap,
            camera_height=h_cam,
            camera_angle=p_cam,
            bev_scale=bev_scale,
            bev_center=bev_center
        )
        loss = self.sync_value(self.loss_fn.forward(pred, batch.gt))

        batch.pred = pred
        batch.loss = self.sync_value(loss)
        batch.size = image.size(0)
        return batch
