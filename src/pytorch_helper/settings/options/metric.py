from dataclasses import dataclass

from .base import OptionBase

__all__ = ['MetricOption']


@dataclass()
class MetricOption(OptionBase):
    ref: str
    kwargs: dict

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = dict()

    def build(self):
        """ build a metric for test
        """
        from ..spaces import Spaces
        return Spaces.build(Spaces.NAME.METRIC, self.ref, self.kwargs)
