from dataclasses import dataclass

from .base import OptionBase

__all__ = ['LossOption']


@dataclass()
class LossOption(OptionBase):
    ref: str
    kwargs: dict

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = dict()

    def build(self):
        """ build loss function
        """
        from ..spaces import Spaces
        return Spaces.build(Spaces.NAME.LOSS_FN, self.ref, self.kwargs)
