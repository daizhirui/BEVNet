import math
import torch
import torch.nn.modules as nn
import torchvision.models as pt_models

from pytorch_helper.settings.spaces import Spaces


@Spaces.register(Spaces.NAME.MODEL, 'PoseNet')
class PoseNet(nn.Module):
    def __init__(self, encoder_type: str, encoder_pretrained=True):
        super(PoseNet, self).__init__()
        self.encoder, left_layers, avgpool_size = build_encoder(
            encoder_type, encoder_pretrained
        )
        # don't freeze the encoder here, otherwise it will be ignored in DDP
        self.encoder.requires_grad_(True)
        self.decoder = PoseDecoder(preceding_layers=left_layers,
                                   avg_output_size=avgpool_size)

    def forward(self, x):
        return self.decoder.forward(self.encoder.forward(x))

    # def train_encoder(self, train=False):
    #     """
    #     This method is effective only during training process
    #     :param train: True to set the encoder trainable
    #     """
    #     train = train and self.training
    #     if not (self.encoder_pretrained or train):
    #         logger.warn("The encoder is not pretrained but it will be frozen")
    #     self.encoder.requires_grad_(train)
    #     self.encoder.train(train)


def build_encoder(encoder_type, encoder_pretrained):
    backbone_builder = getattr(pt_models, encoder_type, None)
    if backbone_builder is None:
        raise ValueError(f'{encoder_type} is unavailable')

    backbone = backbone_builder(encoder_pretrained)

    if backbone_builder in [pt_models.vgg16, pt_models.vgg16_bn]:
        n_layers = 23 if backbone_builder is pt_models.vgg16 else 33
        encoder = backbone.features[:n_layers]
        # the encoder may be shared with other branches
        left_layers = backbone.features[n_layers:]
        avgpool_size = (2, 2)  # output shape (B, 512, H/32, W/32)
    elif backbone_builder in [pt_models.resnet50, pt_models.resnet101]:
        encoder = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3,
        )
        left_layers = [backbone.layer4]  # output shape (B, 2048, H/32, W/32)
        avgpool_size = (1, 1)
    elif backbone_builder is pt_models.inception_v3:
        encoder = nn.Sequential(
            backbone.Conv2d_1a_3x3,
            backbone.Conv2d_2a_3x3,
            backbone.Conv2d_2b_3x3,
            backbone.maxpool1,
            backbone.Conv2d_3b_1x1,
            backbone.Conv2d_4a_3x3,
            backbone.maxpool2,
            backbone.Mixed_5b,
            backbone.Mixed_5c,
            backbone.Mixed_5d
        )
        left_layers = [
            backbone.Mixed_6a,
            backbone.Mixed_6b,
            backbone.Mixed_6c,
            backbone.Mixed_6d,
            backbone.Mixed_7a,
            backbone.Mixed_7b,
            backbone.Mixed_7c,  # output shape (B, 2048, ~H/32, ~W/32)
        ]
        avgpool_size = (1, 1)
    else:
        raise ValueError("Unknown encoder type: {}".format(type(backbone)))

    return encoder, left_layers, avgpool_size


class PoseDecoder(nn.Module):
    def __init__(self, preceding_layers=None, in_channels=2048,
                 avg_output_size=(1, 1)
                 ):
        super(PoseDecoder, self).__init__()
        layers = []
        if preceding_layers is not None:
            layers.extend(preceding_layers)
        layers.append(nn.AdaptiveAvgPool2d(avg_output_size))
        self.bridge = nn.Sequential(*layers)
        self.fc_camera_height = self._make_fc(in_channels)
        self.fc_camera_angle = self._make_fc(in_channels)

    def forward(self, x):
        x = self.bridge(x)
        camera_height = self.fc_camera_height(x)
        camera_angle = math.pi / 2 * torch.sigmoid(self.fc_camera_angle(x))
        return dict(
            camera_height=camera_height.flatten(),
            camera_angle=camera_angle.flatten()
        )

    @staticmethod
    def _make_fc(in_channels=2048, dropout_p=0.5, relu=nn.ReLU):
        return nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_p),
            nn.Linear(in_channels, 512),
            relu(inplace=True),  # nn.Dropout(p=dropout_p),
            nn.Linear(512, 128),
            relu(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 1)
        )
