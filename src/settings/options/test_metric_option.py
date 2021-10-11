from dataclasses import dataclass

from pytorch_helper.settings.options import MetricOption
from pytorch_helper.settings.options import OptionBase

__all__ = ['TestMetricOption']


@dataclass()
class TestMetricOption(OptionBase):
    metrics: list
    save_im: bool = False
    save_model_output: bool = True
    label_risk: bool = False
    test_all: bool = False
    cc_oracle: bool = False
    pose_oracle: bool = False
    iv_map: str = 'head_map'

    def __post_init__(self):
        assert self.metrics is not None and len(self.metrics) > 0, \
            'Please specify at least one metric file'
        self.metrics = [
            MetricOption.from_file(m) for m in self.metrics
        ]
