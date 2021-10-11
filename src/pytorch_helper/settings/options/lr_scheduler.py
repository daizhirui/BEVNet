from dataclasses import dataclass

from .base import OptionBase

__all__ = ['LRSchedulerOption']


@dataclass()
class LRSchedulerOption(OptionBase):
    enable: bool
    ref: str
    kwargs: dict
    metric_key: str = 'loss'

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = dict()

    def build(self, optimizer):
        """ build a learning rate scheduler with the given optimizer

        :param optimizer: Optimizer used to build the learning rate scheduler
        :return: a learning rate scheduler
        """
        if not self.enable:
            return None

        import torch
        builder = getattr(torch.optim.lr_scheduler, self.ref)
        return builder(optimizer, **self.kwargs)
