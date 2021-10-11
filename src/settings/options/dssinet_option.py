from dataclasses import dataclass

import torch

from models.dssinet.CRFVGG import CRFVGG
from models.dssinet.network import load_net
from pytorch_helper.settings.options import ModelOption
from pytorch_helper.settings.options import TaskOption
from pytorch_helper.settings.options.descriptors import AutoConvertDescriptor
from pytorch_helper.settings.spaces import Spaces
from pytorch_helper.utils.log import get_logger
from .test_metric_option import TestMetricOption

__all__ = ['DSSINetTaskOption', 'DSSINetOption']

logger = get_logger(__name__)


@dataclass()
class DSSINetOption(ModelOption):
    load_from_pretrained: bool

    def build(self):
        model = CRFVGG()
        logger.info(f'Build {type(model).__name__}')

        state_dict = None
        if self.pth_available:
            if self.load_from_pretrained:
                load_net(self.pth_path, model, prefix='model.module.')
                logger.info(f'Load model state from {self.pth_path}')
            else:
                state_dict = torch.load(self.pth_path)
                model.load_state_dict(state_dict['model'])
                logger.info(
                    f'Load model state from epoch {state_dict["epoch"]}')
        else:
            logger.warn('No pth file to load the model state, double check!')
        return model, state_dict


@dataclass()
class DSSINetTaskOptionData(TaskOption):
    pose_net: ModelOption = None


@Spaces.register(Spaces.NAME.TASK_OPTION, ['DSSINetTask', 'DSSINet2BEVTask'])
class DSSINetTaskOption(DSSINetTaskOptionData):
    model = AutoConvertDescriptor(DSSINetOption.from_dict)
    pose_net = AutoConvertDescriptor(ModelOption.from_dict)
    test_option = AutoConvertDescriptor(TestMetricOption.from_dict)
