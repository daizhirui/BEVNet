from dataclasses import dataclass
from typing import List
from typing import Optional
from typing import Union

from .base import OptionBase
from .train_routine import TrainRoutine

__all__ = ['TrainSettingOption']


@dataclass()
class TrainSettingOption(OptionBase):
    start_epoch: int
    epochs: int
    save_model_freq: int
    valid_on_test: bool
    train_routines: List[Union[dict, TrainRoutine]]
    gradient_clip: float = 0
    detect_gradient_explosion: bool = False
    gradient_explosion_threshold: float = 1e5

    def __post_init__(self):
        self.train_routines = [
            TrainRoutine(**r) for r in self.train_routines
        ]
        self.train_routines.sort(key=lambda x: x.epochs)

    def get_train_routine(self, epoch: int) -> Optional[TrainRoutine]:
        """ find the train routine for the specified epoch

        :param epoch: int of epoch number
        :return: TrainRoutine or None
        :raise LookupError: no train routine available for the specified epoch
            when `self.train_routines` is not None
        """
        if self.train_routines is None:
            return
        last_epoch = 0
        for r in self.train_routines:
            if epoch < r.epochs:
                r.new_routine = epoch == last_epoch
                r.optimizer_reset = r.optimizer_reset and r.new_routine
                return r
            else:
                last_epoch = r.epochs
        raise LookupError(f'cannot find train routine for epoch {epoch}')
