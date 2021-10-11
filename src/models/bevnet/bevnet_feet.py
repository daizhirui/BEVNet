from .bev_decoder import BEVDecoder
from .bevnet_base import BEVNetBase
from pytorch_helper.settings.spaces import Spaces

__all__ = ['BEVNetFeetOnly']


@Spaces.register(Spaces.NAME.MODEL, 'BEVNetFeetOnly')
class BEVNetFeetOnly(BEVNetBase):

    def __init__(self, feet_branch, pose_branch, shared_encoder=None,
                 branches_share_encoder=(), magnitude_scale=100
                 ):
        assert feet_branch is not None, \
            f'{BEVNetFeetOnly.__name__} requires feet_branch != None'

        in_channels_feet = self._get_in_channels(
            'feet', feet_branch, shared_encoder, branches_share_encoder
        )

        decoder_bev = BEVDecoder(
            0, in_channels_feet, magnitude_scale
        )

        super(BEVNetFeetOnly, self).__init__(
            head_branch=None,
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
        feet_map = None
        if x_feet_iv is not None:
            feet_map = self.decoder_feet(x_feet_iv)

        cam_params = [h_cam, p_cam, fu / 8, fv / 8]
        x_feet_bev = self.bev_transform(x_feet_iv, 0, *cam_params)[0]

        # decode to bev map
        bev_map = self.decoder_bev(x_feet_bev)
        # bev scale and center
        scales, centers = self.bev_transform.get_bev_scales_and_centers(
            x.size(-1), h_cam, p_cam, fv
        )
        # summarize output
        out = dict(
            feet_map=feet_map,
            bev_map=bev_map,
            bev_scale=scales,
            bev_center=centers,
            **camera_pose
        )

        return out
