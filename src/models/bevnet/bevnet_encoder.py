import torch.nn.modules as nn


class BEVNetEncoder(nn.Module):
    branch_names = ['head', 'feet', 'pose']

    def __init__(self, encoder_head=None, encoder_feet=None, encoder_pose=None,
                 encoder_shared=None, branches_share_encoder=()
                 ):
        super(BEVNetEncoder, self).__init__()
        # these four attributes must be kept
        # in order to save and load weights correctly
        self.encoder_head = encoder_head
        self.encoder_feet = encoder_feet
        self.encoder_pose = encoder_pose
        self.encoder_shared = None
        # translate share_encoder_mode
        self.share_encoder_mode = {
            x: x in branches_share_encoder for x in self.branch_names
        }
        # setup encoder_shared
        if True in self.share_encoder_mode.values():
            assert encoder_shared is not None, \
                f"shared_encoder is required for: {branches_share_encoder}"
            self.encoder_shared = encoder_shared

    def forward(self, x):
        # forward shared encoder
        shared_output = self.encoder_shared(x) if self.encoder_shared else None
        # forward dedicated encoders and generate the output
        out = dict()
        encoders = [self.encoder_head, self.encoder_feet, self.encoder_pose]
        for branch, encoder in zip(self.branch_names, encoders):
            if self.share_encoder_mode[branch]:
                out[branch] = shared_output
            else:
                out[branch] = encoder(x) if encoder else None
        return out
