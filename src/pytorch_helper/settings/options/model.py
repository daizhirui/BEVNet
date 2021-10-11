import os
from dataclasses import dataclass

from .base import OptionBase
from ...utils.log import get_logger

__all__ = ['ModelOption']

logger = get_logger(__name__)


@dataclass()
class ModelOption(OptionBase):
    ref: str
    kwargs: dict
    pth_path: str

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = dict()

    @property
    def pth_available(self):
        """ check if `self.pth_path` is available

        :return: Bool to indicate if the checkpoint file is available
        """
        if self.pth_path:
            if os.path.isfile(self.pth_path):
                return True
            elif os.path.isdir(self.pth_path):
                raise IsADirectoryError(
                    f'pth path "{self.pth_path}" should not be a folder'
                )
            else:
                raise FileNotFoundError(
                    f'{self.pth_path} does not exist'
                )
        return False

    def build(self):
        """ build the model and load the weights from the checkpoint if
        `self.pth_available` is True.

        :return: model and state_dict
        """
        from ..spaces import Spaces
        from ...utils.io import load_pth

        model = Spaces.build(Spaces.NAME.MODEL, self.ref, self.kwargs)
        # model = Spaces.build_model(self.name, **self.kwargs)
        logger.info(f'Build {type(model).__name__}')

        state_dict = None
        if self.pth_available:
            state_dict = load_pth(self.pth_path)
            model.load_state_dict(state_dict['model'])
            logger.info(f'Load model state from {self.pth_path}')
        return model, state_dict
