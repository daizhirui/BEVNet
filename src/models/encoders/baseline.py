import torch.nn.modules as nn

from models.basic_block import BasicConv2d


class EncoderBase(nn.Module):
    def __init__(self, features):
        super(EncoderBase, self).__init__()
        self.features = features

    def forward(self, x):
        return self.features(x)


class EncoderBaseline(EncoderBase):

    def __init__(self):
        super(EncoderBaseline, self).__init__(features=nn.Sequential(
            BasicConv2d(3, 16, kernel_size=9, stride=1, padding=4),
            BasicConv2d(16, 32, kernel_size=7, stride=1, padding=3),
            BasicConv2d(32, 20, kernel_size=7, stride=1, padding=3),
            BasicConv2d(20, 40, kernel_size=5, stride=2, padding=2),  # 1/2
            BasicConv2d(40, 20, kernel_size=5, stride=2, padding=2),  # 1/4
            BasicConv2d(20, 10, kernel_size=5, stride=1, padding=2),
        ))
