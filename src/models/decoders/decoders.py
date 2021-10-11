import torch.nn.modules as nn

from models.basic_block import BasicConv2d
from models.basic_block import BasicDeconv2d


class DecoderBase(nn.Module):

    def __init__(self, decode_block, output_block, magnitude_scale=100):
        super(DecoderBase, self).__init__()
        self.decoded_feature = None
        self.decode_block = decode_block
        self.output_block = output_block
        self.magnitude_scale = magnitude_scale

    def forward(self, x, keep_decoded_feature=False):
        self.decoded_feature = self.decode_block(x)
        out = self.output_block(self.decoded_feature)
        if not keep_decoded_feature:
            self.decoded_feature = None
        return out


class DecoderVGG16(DecoderBase):
    # the output of VGG16/VGG16bn encoder is of 512 channels
    def __init__(self, magnitude_scale=100):
        super(DecoderVGG16, self).__init__(
            decode_block=nn.Sequential(
                BasicConv2d(512, 256),
                BasicDeconv2d(256, 128),  # x2
                BasicConv2d(128, 64),
                BasicDeconv2d(64, 32),  # x4
                BasicConv2d(32, 16),
                BasicDeconv2d(16, 8)  # x8
            ),
            output_block=nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),
            magnitude_scale=magnitude_scale
        )


class DecoderResNet(DecoderBase):
    # the output of ResNet50/ResNet101 encoder is of 1024 channels
    def __init__(self, magnitude_scale=100):
        super(DecoderResNet, self).__init__(
            decode_block=nn.Sequential(
                BasicConv2d(1024, 512),
                BasicDeconv2d(512, 256),  # x2
                BasicConv2d(256, 128),
                BasicDeconv2d(128, 64),  # x4
                BasicConv2d(64, 32),
                BasicDeconv2d(32, 16)  # x8
            ),
            output_block=nn.Sequential(
                BasicConv2d(16, 8),
                nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1)
            ),
            magnitude_scale=magnitude_scale
        )


class DecoderInception3(DecoderBase):
    # the output of Inception3 encoder is of 288 channels
    def __init__(self, magnitude_scale=100):
        super(DecoderInception3, self).__init__(
            decode_block=nn.Sequential(
                BasicConv2d(288, 256),
                BasicDeconv2d(256, 128),  # x2
                BasicConv2d(128, 64),
                BasicDeconv2d(64, 32),  # x4
                BasicConv2d(32, 16),
                BasicDeconv2d(16, 8)  # x8
            ),
            output_block=nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),
            magnitude_scale=magnitude_scale
        )


class DecoderBaseline(DecoderBase):

    def __init__(self, magnitude_scale=100):
        super(DecoderBaseline, self).__init__(
            decode_block=nn.Sequential(
                BasicConv2d(10, 20),
                BasicConv2d(20, 32),
                BasicDeconv2d(32, 16, kernel_size=3, stride=2, padding=1,
                              output_padding=1),
                BasicDeconv2d(16, 8, kernel_size=3, stride=2, padding=1,
                              output_padding=1),
            ),
            output_block=nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),
            magnitude_scale=magnitude_scale
        )
