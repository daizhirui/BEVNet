import torch

from .bev_decoder import BEVDecoder
from .bevnet_base import BEVNetBase
from pytorch_helper.settings.spaces import Spaces

__all__ = ['BEVNetNoAttention']


class HumanAvg:
    """
    Average human dimension, each constant is the average of male and female
    average results.
    Unit: meter
    """
    # https://www.firstinarchitecture.co.uk/average-male-and-female-dimensions/
    # https://en.wikipedia.org/wiki/Human_head
    AVG_SHOULDER_WIDTH = (465 + 395) / 2000.
    AVG_WAIST_WIDTH = (360 + 370) / 2000.
    AVG_MALE_HEIGHT = 1740 / 1000.
    AVG_FEMALE_HEIGHT = 1610 / 1000.
    AVG_HEIGHT = (1740 + 1610) / 2000.
    AVG_EYE_HEIGHT = (1630 + 1505) / 2000.
    AVG_HEAD_WIDTH = (165 + 158) / 2000.


@Spaces.register(Spaces.NAME.MODEL, 'BEVNetNoAttention')
class BEVNetNoAttention(BEVNetBase):

    def __init__(self, head_branch, feet_branch, pose_branch,
                 shared_encoder=None, branches_share_encoder=(),
                 magnitude_scale=100
                 ):
        assert head_branch is not None, \
            f"head branch is required for {type(self).__name__}, " \
            f"without head branch, please use BEVNetFeetOnly"

        in_channels_head = self._get_in_channels(
            'head', head_branch, shared_encoder, branches_share_encoder
        )
        in_channels_feet = self._get_in_channels(
            'feet', feet_branch, shared_encoder, branches_share_encoder
        )

        decoder_bev = BEVDecoder(
            in_channels_head, in_channels_feet, magnitude_scale
        )

        super(BEVNetNoAttention, self).__init__(
            head_branch=head_branch,
            feet_branch=feet_branch,
            pose_branch=pose_branch,
            decoder_bev=decoder_bev,
            shared_encoder=shared_encoder,
            branches_share_encoder=branches_share_encoder
        )

    def forward(self, x, fu, fv=None, h_cam=None, p_cam=None):
        out = self.encoder(x)
        x_head_iv, x_feet_iv, x_pose = out['head'], out['feet'], out['pose']

        camera_pose = self.decoder_pose(x_pose)
        if h_cam is None or p_cam is None:
            h_cam = camera_pose['camera_height']
            p_cam = camera_pose['camera_angle']

        # image view
        head_map = self.decoder_head(x_head_iv)
        feet_map = None
        if x_feet_iv is not None:
            feet_map = self.decoder_feet(x_feet_iv)

        # transform features to bev
        cam_params = [h_cam, p_cam, fu / 8, fv / 8]
        h = HumanAvg.AVG_EYE_HEIGHT
        decoder_bev_in = [
            self.bev_transform(x_head_iv, h, *cam_params)[0]
        ]

        if x_feet_iv is not None:
            x_feet_bev = self.bev_transform(x_feet_iv, 0, *cam_params)[0]
            decoder_bev_in.append(x_feet_bev)
        x_bev = torch.cat(decoder_bev_in, dim=1)

        # decode to bev map
        bev_map = self.decoder_bev(x_bev)
        # bev scale and center
        scales, centers = self.bev_transform.get_bev_scales_and_centers(
            x.size(-1), h_cam, p_cam, fv
        )
        # summarize output
        out = dict(
            head_map=head_map,
            feet_map=feet_map,
            bev_map=bev_map,
            bev_scale=scales,
            bev_center=centers,
            **camera_pose
        )

        return out
