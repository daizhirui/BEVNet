import torch.nn.modules as nn
from torchvision.models import resnet50

from .baseline import EncoderBase


class EncoderResNet(EncoderBase):

    def __init__(self, resnet, pretrained=True):
        loaded_net = resnet(pretrained)
        if resnet is resnet50:
            layer3_blocks = 6
        else:
            layer3_blocks = 23
        layer3 = self.make_layer3(layer3_blocks, loaded_net)

        super(EncoderResNet, self).__init__(features=nn.Sequential(
            loaded_net.conv1,
            loaded_net.bn1,
            loaded_net.relu,
            loaded_net.maxpool,
            loaded_net.layer1,
            loaded_net.layer2,
            layer3
        ))

    @staticmethod
    def make_layer3(blocks, resnet=None):
        block = Bottleneck
        in_planes = 512
        planes = 256
        downsample = nn.Sequential(
            conv1x1(in_planes, planes * block.expansion, stride=1),
            nn.BatchNorm2d(planes * block.expansion),
        )

        layers = [
            block(in_planes, planes, stride=1, downsample=downsample,
                  norm_layer=nn.BatchNorm2d)
        ]
        in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(in_planes, planes, norm_layer=nn.BatchNorm2d))

        layer3 = nn.Sequential(*layers)
        layer3.load_state_dict(resnet.layer3.state_dict())
        return layer3


# source code from torchvision.models.resnet
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for down-sampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None
                 ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
