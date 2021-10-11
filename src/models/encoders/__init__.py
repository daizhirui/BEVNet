from torchvision.models import resnet101
from torchvision.models import resnet50
from torchvision.models import vgg16
from torchvision.models import vgg16_bn

from .baseline import EncoderBaseline
from .inception import EncoderInception3
from .resnet import EncoderResNet
from .vgg16 import EncoderVGG16

encoder_builders = {
    'baseline': lambda x: EncoderBaseline(),
    'inception_v3': EncoderInception3,
    'resnet50': lambda pretrained: EncoderResNet(resnet50, pretrained),
    'resnet101': lambda pretrained: EncoderResNet(resnet101, pretrained),
    'vgg16': lambda pretrained: EncoderVGG16(vgg16, 23, pretrained),
    'vgg16_bn': lambda pretrained: EncoderVGG16(vgg16_bn, 33, pretrained)
}


def build_encoder(encoder_type, *args, **kwargs):
    return encoder_builders[encoder_type](*args, **kwargs)
