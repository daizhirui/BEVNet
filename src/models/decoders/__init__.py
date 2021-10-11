from .decoders import DecoderBaseline
from .decoders import DecoderInception3
from .decoders import DecoderResNet
from .decoders import DecoderVGG16

decoder_builders = {
    'baseline': DecoderBaseline,
    'vgg16': DecoderVGG16,
    'vgg16_bn': DecoderVGG16,
    'resnet50': DecoderResNet,
    'resnet101': DecoderResNet,
    'inception_v3': DecoderInception3
}


def build_decoder(encoder_type, **kwargs):
    return decoder_builders[encoder_type](**kwargs)
