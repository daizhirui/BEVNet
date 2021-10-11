from .baseline import EncoderBase


class EncoderVGG16(EncoderBase):

    def __init__(self, vgg, n_layers, pretrained=True):
        super(EncoderVGG16, self).__init__(
            features=vgg(pretrained).features[:n_layers])
