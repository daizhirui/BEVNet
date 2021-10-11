import torch.nn.modules as nn
from pytorch_helper.utils.log import get_logger

from models.decoders import build_decoder
from models.encoders import build_encoder
from pytorch_helper.settings.spaces import Spaces
logger = get_logger(__name__)


@Spaces.register(Spaces.NAME.MODEL, 'IVNet')
class IVNet(nn.Module):

    def __init__(self, encoder_type, encoder_pretrained, iv_map,
                 magnitude_scale=100
                 ):
        super(IVNet, self).__init__()
        self.iv_map = iv_map
        # build encoder
        self.encoder_pretrained = encoder_pretrained
        self.encoder = build_encoder(encoder_type, encoder_pretrained)
        logger.info(f'Build encoder: {encoder_type} (pretrained: '
                    f'{encoder_pretrained})')
        # should not freeze encoder here, otherwise it will be ignored in DDP
        self.encoder.requires_grad_(True)
        # build decoder
        self.decoder = build_decoder(
            encoder_type, magnitude_scale=magnitude_scale
        )
        logger.info(
            f'Build decoder: {type(self.decoder).__name__} '
            f'(magnitude_scale={magnitude_scale})'
        )

    def forward(self, x, keep_decoded_feature=False):
        return self.decoder.forward(
            self.encoder.forward(x), keep_decoded_feature
        )
