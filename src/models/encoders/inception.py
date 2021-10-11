import torch.nn.modules as nn
from torchvision.models.inception import inception_v3

from .baseline import EncoderBase


class EncoderInception3(EncoderBase):

    def __init__(self, pretrained=True, same_padding=True):
        """
        :param pretrained: if True, load weights from pretrained Inception v3 model
        :param same_padding: if True, add padding to non-down-sample layers to make the output
                             has the same size as the input's.
        """
        super(EncoderInception3, self).__init__(features=None)
        loaded_net = inception_v3(pretrained=pretrained, aux_logits=False,
                                  transform_input=False, init_weights=False)

        conv2d_1a_3x3 = loaded_net.Conv2d_1a_3x3
        conv2d_2a_3x3 = loaded_net.Conv2d_2a_3x3
        conv2d_2b_3x3 = loaded_net.Conv2d_2b_3x3
        maxpool1 = loaded_net.maxpool1
        conv2d_3b_1x1 = loaded_net.Conv2d_3b_1x1
        conv2d_4a_3x3 = loaded_net.Conv2d_4a_3x3
        maxpool2 = loaded_net.maxpool2
        mixed_5b = loaded_net.Mixed_5b
        mixed_5c = loaded_net.Mixed_5c
        mixed_5d = loaded_net.Mixed_5d

        if same_padding:
            conv2d_1a_3x3.conv.padding = 1
            conv2d_2a_3x3.conv.padding = 1
            maxpool1.padding = 1
            conv2d_4a_3x3.conv.padding = 1
            maxpool2.padding = 1

        self.features = nn.Sequential(
            conv2d_1a_3x3,
            conv2d_2a_3x3, conv2d_2b_3x3,
            maxpool1, conv2d_3b_1x1,
            conv2d_4a_3x3,
            maxpool2, mixed_5b, mixed_5c, mixed_5d
        )
