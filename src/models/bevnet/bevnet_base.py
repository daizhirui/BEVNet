import torch.nn.modules as nn

from models.encoders import EncoderBaseline
from models.encoders import EncoderInception3
from models.encoders import EncoderResNet
from models.encoders import EncoderVGG16
from .bev_transform import BEVTransform
from .bevnet_encoder import BEVNetEncoder


class BEVNetBase(nn.Module):
    branch_names = ['head', 'feet', 'bev', 'pose']
    encoder_out_channels = {
        EncoderVGG16: 512,
        EncoderResNet: 1024,
        EncoderInception3: 288,
        EncoderBaseline: 10
    }

    def __init__(self, head_branch, feet_branch, pose_branch, decoder_bev,
                 shared_encoder=None, branches_share_encoder=()
                 ):
        super(BEVNetBase, self).__init__()
        self.encoder = BEVNetEncoder(
            encoder_head=head_branch.encoder if head_branch else None,
            encoder_feet=feet_branch.encoder if feet_branch else None,
            encoder_pose=pose_branch.encoder,
            encoder_shared=shared_encoder,
            branches_share_encoder=branches_share_encoder
        )

        self.decoder_bev = decoder_bev
        self.decoder_feet = feet_branch.decoder if feet_branch else None
        self.decoder_head = head_branch.decoder if head_branch else None
        self.decoder_pose = pose_branch.decoder

        self.bev_transform = BEVTransform()

        self.available_branches = ['pose', 'bev']
        if self.decoder_head is not None:
            self.available_branches.append('head')
        if self.decoder_feet is not None:
            self.available_branches.append('feet')

        # rewire train_encoder function
        # self.train_encoder = self.encoder.train_encoder

    def forward(self, x, fu, fv=None, h_cam=None, p_cam=None):
        raise NotImplementedError

    @staticmethod
    def _get_in_channels(name, branch, shared_encoder, branches_share_encoder):
        if shared_encoder and name in branches_share_encoder:
            return BEVNetBase.encoder_out_channels[type(shared_encoder)]
        elif branch is None:
            return 0
        else:
            return BEVNetBase.encoder_out_channels[type(branch.encoder)]
