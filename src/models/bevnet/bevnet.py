import torch
import torch.nn as nn
from torch.nn.functional import softmax

from .bev_decoder import BEVDecoder
from .bevnet_base import BEVNetBase
from ..basic_block import BasicConv2d

from pytorch_helper.settings.spaces import Spaces


@Spaces.register(Spaces.NAME.MODEL, 'BEVNet')
class BEVNet(BEVNetBase):

    def __init__(self, head_branch, feet_branch, pose_branch,
                 shared_encoder=None, branches_share_encoder=(),
                 magnitude_scale=100, head_heights=()
                 ):
        assert head_branch is not None, \
            f"head branch is required for {type(self).__name__}, " \
            f"without head branch, please use BEVNetFeetOnly"
        assert len(head_heights) > 1, 'len(head_heights) should >= 2'

        in_channels_head = self._get_in_channels(
            'head', head_branch, shared_encoder, branches_share_encoder
        )
        in_channels_feet = self._get_in_channels(
            'feet', feet_branch, shared_encoder, branches_share_encoder
        )

        decoder_bev = BEVDecoder(
            in_channels_head, in_channels_feet, magnitude_scale
        )

        super(BEVNet, self).__init__(
            head_branch=head_branch,
            feet_branch=feet_branch,
            pose_branch=pose_branch,
            decoder_bev=decoder_bev,
            shared_encoder=shared_encoder,
            branches_share_encoder=branches_share_encoder
        )

        self.head_heights = head_heights

        self.head_height_attention = nn.Sequential(
            BasicConv2d(in_channels_head, in_channels_head // 2, kernel_size=1,
                        stride=1, padding=0),
            BasicConv2d(in_channels_head // 2, in_channels_head // 2,
                        kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels_head // 2, in_channels_head // 4,
                        kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_channels_head // 4, in_channels_head // 4,
                        kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels_head // 4, 1, kernel_size=1, stride=1,
                        padding=0)
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

        # transform features to bev with head attention mechanism
        x_head_bev = []
        height_attn = []
        cam_params = [h_cam, p_cam, fu / 8, fv / 8]
        for h in self.head_heights:
            x_bev = self.bev_transform(x_head_iv, h, *cam_params)[0]
            x_attn = self.head_height_attention(x_bev)
            x_head_bev.append(x_bev.unsqueeze(1))
            height_attn.append(x_attn)
        height_attn = torch.cat(height_attn, dim=1)
        height_attn = softmax(height_attn, dim=1)  # (N, M, H, W)
        x_head_bev = torch.cat(x_head_bev, dim=1)  # (N, M, C, H, W)
        x_head_bev = (x_head_bev * height_attn.unsqueeze(2)).sum(dim=1)
        decoder_bev_in = [x_head_bev]

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
            height_attn=height_attn,
            **camera_pose
        )

        return out
