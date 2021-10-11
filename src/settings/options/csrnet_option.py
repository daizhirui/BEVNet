from dataclasses import dataclass

import torch

from models.csrnet import CSRNet
from pytorch_helper.settings.options import ModelOption
from pytorch_helper.settings.options import TaskOption
from pytorch_helper.settings.options.descriptors import AutoConvertDescriptor
from pytorch_helper.settings.spaces import Spaces
from pytorch_helper.utils.log import get_logger
from .test_metric_option import TestMetricOption

__all__ = ['CSRNetTaskOption', 'CSRNetOption']

logger = get_logger(__name__)


@dataclass()
class CSRNetOption(ModelOption):
    load_from_pretrained: bool

    def build(self):
        model = CSRNet()
        logger.info(f'Build {type(model).__name__}')

        state_dict = None
        if self.pth_available:
            state_dict = torch.load(self.pth_path, torch.device('cpu'))
            if self.load_from_pretrained:
                for k, v in model.state_dict().items():
                    k = f'CCN.{k}'
                    v.copy_(state_dict[k])
                    logger.info(f'Load {k}')
                logger.info(f'Load model state from {self.pth_path}')
                state_dict = None
            else:
                model.load_state_dict(state_dict['model'])
                logger.info(
                    f'Load model state from epoch {state_dict["epoch"]}')
        else:
            logger.warn('No pth file to load the model state, double check!')
        return model, state_dict


@dataclass()
class CSRNetTaskOptionData(TaskOption):
    pose_net: ModelOption = None


@Spaces.register(Spaces.NAME.TASK_OPTION, ['CSRNetTask', 'CSRNet2BEVTask'])
class CSRNetTaskOption(CSRNetTaskOptionData):
    model = AutoConvertDescriptor(CSRNetOption.from_dict)
    pose_net = AutoConvertDescriptor(ModelOption.from_dict)
    test_option = AutoConvertDescriptor(TestMetricOption.from_dict)
