from dataclasses import dataclass

from .base import OptionBase

__all__ = ['DataloaderOption']


@dataclass()
class DataloaderOption(OptionBase):
    ref: str
    kwargs: dict

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = dict()

    def build(self):
        """ build a dataloader
        """
        from ..spaces import Spaces

        return Spaces.build(Spaces.NAME.DATALOADER, self.ref, self.kwargs)
