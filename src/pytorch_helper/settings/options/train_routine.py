from dataclasses import dataclass
from typing import Union

from .base import OptionBase
from ...utils.log import get_logger

__all__ = ['TrainRoutine']

logger = get_logger(__name__)


@dataclass()
class TrainRoutine(OptionBase):
    epochs: int
    init_lr: float = None
    new_routine: bool = True
    train_modules: Union[list, str] = None
    optimizer_reset: bool = True
    note: str = None

    def __post_init__(self):
        if self.train_modules is not None:
            if isinstance(self.train_modules, str):
                self.train_modules = [
                    x.strip() for x in self.train_modules.strip().split(',')
                ]

    def set_init_lr(self, optimizer):
        """ set the learning rate of the optimizer if `self` is a new routine
        and `init_lr` is not None.

        :param optimizer: the optimizer to set its learning rate
        :return: Bool to indicate if learning rate is set
        """
        if not self.new_routine:
            return False
        self.new_routine = False
        if self.init_lr is None:
            return False
        logger.info(f'set init-lr to {self.init_lr} for new routine')
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.init_lr
            param_group['init_lr'] = self.init_lr
        return True
