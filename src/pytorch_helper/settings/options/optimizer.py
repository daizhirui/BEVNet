from dataclasses import dataclass

import torch

from .base import OptionBase

__all__ = ['OptimizerOption']


@dataclass()
class OptimizerOption(OptionBase):
    ref: str
    kwargs: dict

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = dict()

    def build(self, model):
        """ build the optimizer with the given model

        :param model: the model used to build the optimizer, only parameters
            whose `requires_grad` is True will be posted to the optimizer
        :return: the optimizer
        """
        builder = getattr(torch.optim, self.ref)
        return builder(
            filter(lambda p: p.requires_grad, model.parameters()), **self.kwargs
        )
