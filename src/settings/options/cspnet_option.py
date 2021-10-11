from dataclasses import dataclass
from typing import Union

import torch
from pytorch_helper.data.transform import DeNormalize
from pytorch_helper.settings.options import ModelOption
from pytorch_helper.settings.options import TaskOption
from pytorch_helper.utils.log import get_logger
from torchvision.transforms import Normalize
from pytorch_helper.settings.spaces import Spaces
from pytorch_helper.settings.options.descriptors import AutoConvertDescriptor
from datasets.cityuhk.dataset import CityUHKBEV
from models import CSPNet
from .test_metric_option import TestMetricOption

__all__ = ['CSPNetOption', 'CSPNetTaskOption']

logger = get_logger(__name__)


@dataclass()
class CSPNetOption(ModelOption):
    load_from_pretrained: bool

    def build(self):
        model = CSPNet()
        logger.info(f'Build {type(model).__name__}')

        state_dict = None
        if self.pth_available:
            state_dict = torch.load(self.pth_path)
            if self.load_from_pretrained:
                model.load_state_dict(state_dict)
                logger.info(f'Load model state from {self.pth_path}')
            else:
                model.load_state_dict(state_dict['model'])
                logger.info(
                    f'Load model state from epoch {state_dict["epoch"]}')
        else:
            logger.warn('No pth file to load the model state, double check!')
        return model, state_dict


@dataclass()
class CSPNetTaskOptionData(TaskOption):
    pose_net: ModelOption = None


@Spaces.register(Spaces.NAME.TASK_OPTION, 'CSPNet2BEVTask')
class CSPNetTaskOption(CSPNetTaskOptionData):
    model = AutoConvertDescriptor(CSPNetOption.from_dict)
    pose_net = AutoConvertDescriptor(ModelOption.from_dict)
    test_option = AutoConvertDescriptor(TestMetricOption.from_dict)

    def __post_init__(self, mode, is_distributed):
        CityUHKBEV.normalize = Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        CityUHKBEV.denormalize = DeNormalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        super(CSPNetTaskOption, self).__post_init__(mode, is_distributed)
