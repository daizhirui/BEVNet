import torch.nn.modules as nn

from models.basic_block import BasicConv2d
from models.basic_block import BasicDeconv2d
from models.decoders.decoders import DecoderBase


class BEVDecoder(DecoderBase):

    def __init__(self, in_channels_head, in_channels_feet, magnitude_scale=100):
        in_channels = in_channels_head + in_channels_feet
        assert in_channels > 64, \
            f'the total of input channels `{in_channels}` is fewer than 64'

        decode_layers = [
            nn.BatchNorm2d(in_channels),
            # normalize the transformed feature map
            BasicConv2d(in_channels, in_channels),
            BasicConv2d(in_channels, in_channels, kernel_size=1, padding=0)
            # channel re-weight
        ]

        for _ in range(3):  # up-sample and channel reduction
            decode_layers.extend([
                BasicConv2d(in_channels, in_channels // 2),
                BasicDeconv2d(in_channels // 2, in_channels // 4),
                BasicConv2d(in_channels // 4, in_channels // 4,
                            kernel_size=1, padding=0)  # channel re-weight
            ])
            in_channels //= 4

        while in_channels > 8:
            decode_layers.append(BasicConv2d(in_channels, in_channels // 2))
            in_channels //= 2

        decode_block = nn.Sequential(*decode_layers)
        output_block = nn.Conv2d(in_channels, 1, 3, stride=1, padding=1)

        super(BEVDecoder, self).__init__(
            decode_block, output_block, magnitude_scale
        )

        self.decode_head_branch = in_channels_head > 0
        self.decode_feet_branch = in_channels_feet > 0
