from enum import Enum
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Collection
from typing import Dict
from typing import Final
from typing import Iterable

import numpy as np

from .io import load_from_pickle
from .io import save_as_pickle
from .. import __version__

__all__ = ['Meter']


class MeterItem:
    class RecordOp(Enum):
        APPEND = 0
        EXTEND = 1

    class ReduceOp(Enum):
        STORE = 0
        SUM = 1

    def __init__(self, record_op: RecordOp, reduce_op: ReduceOp):
        self.record_op: Final = record_op
        self.reduce_op: Final = reduce_op
        self.record: Final[Callable] = self._store_record \
            if reduce_op == self.ReduceOp.STORE else self._running_sum_record

        self.data = [] if reduce_op == self.ReduceOp.STORE else 0.
        self.value_cnt = 0
        self.weight_cnt = 0

    def state_dict(self):
        return dict(
            record_op=self.record_op,
            reduce_op=self.reduce_op,
            data=self.data,
            value_cnt=self.value_cnt,
            weight_cnt=self.weight_cnt,
            version=__version__  # reserved for future use
        )

    def load_state_dict(self, state_dict):
        for key in self.state_dict():
            if key == 'version':
                continue
            setattr(self, key, state_dict[key])

    def _pre_process(self, value, weight):
        if self.record_op == self.RecordOp.APPEND:
            value = [value]
        if not isinstance(weight, Collection):
            weight = [weight] * len(value)
        return value, weight

    def _store_record(self, value: Any, weight=1):
        value, weight = self._pre_process(value, weight)

        for w, v in zip(weight, value):
            self.data.append(np.array(v) * w)
            self.weight_cnt += w
        self.value_cnt = len(self.data)

    def _running_sum_record(self, value: Any, weight=1):
        value, weight = self._pre_process(value, weight)

        for w, v in zip(weight, value):
            self.data += np.array(v, dtype=float) * w
            self.value_cnt += 1
            self.weight_cnt += w

    def sum(self):
        """ get the sum of all the recorded values.

        :return: sum
        """
        if self.reduce_op == self.ReduceOp.STORE:
            return np.sum(np.stack(self.data), axis=0)
        else:
            return self.data

    def mean(self):
        """ get the mean of all the recorded values

        :return: mean
        """
        return self.sum() / self.weight_cnt

    def reset(self):
        self.data = [] if self.reduce_op == self.ReduceOp.STORE else 0.
        self.value_cnt = 0
        self.weight_cnt = 0


class Meter(object):
    RecordOp: ClassVar = MeterItem.RecordOp
    ReduceOp: ClassVar = MeterItem.ReduceOp

    def __init__(self):
        """ Meter is designed for tracking average and sum
        """
        self.meter_items: Dict[str, MeterItem] = dict()

    def __getitem__(self, tag: str) -> MeterItem:
        return self.meter_items[tag]

    def __contains__(self, tag: str) -> bool:
        return tag in self.meter_items

    def _delete_tag(self, tag: str):
        if tag in self.meter_items:
            del self.meter_items[tag]

    def reset(self, tag: str = None):
        """ remove the data of tag

        :param tag: str of tag, if None, remove all the data
        """
        if tag is None:
            tags = self.meter_items.keys()
        else:
            tags = [tag]
        for tag in tags:
            if tag in self.meter_items:
                self.meter_items[tag].reset()

    def reset_tags(self, tags: Iterable[str] = None):
        """ apply `self.reset` on each element in `tags`

        :param tags: Iterable of tags, if None, remove all the data
        """
        if tags is None:
            tags = self.meter_items.keys()
        for tag in tags:
            if tag in self.meter_items:
                self.meter_items[tag].reset()

    def record(
        self, tag: str, value: Any, weight=1, record_op: RecordOp = None,
        reduce_op: ReduceOp = None
    ):
        if tag not in self.meter_items:
            assert record_op is not None, \
                f'Need record_op to create a new {MeterItem.__name__}'
            assert reduce_op is not None, \
                f'Need reduce_op to create a new {MeterItem.__name__}'
            self.meter_items[tag] = MeterItem(record_op, reduce_op)
        meter = self.meter_items[tag]
        meter.record(value, weight)

    def mean(self, tag: str):
        return self.meter_items[tag].mean()

    def means(self, tags: Iterable[str] = None):
        if tags is None:
            tags = self.meter_items.keys()
        return {tag: self.meter_items[tag].mean() for tag in tags}

    def state_dict(self):
        return dict(
            meter_items={
                k: v.state_dict() for k, v in self.meter_items.items()
            },
            version=__version__
        )

    def load_state_dict(self, state_dict):
        for tag, state in state_dict['meter_items'].items():
            item = MeterItem(state['record_op'], state['reduce_op'])
            item.load_state_dict(state)
            self.meter_items[tag] = item

    def save(self, path: str):
        """ save `self.data` and `self.cnt` to `path` as a pickle file

        :param path: str of pickle file path
        """
        save_as_pickle(path, self.state_dict())

    @staticmethod
    def load(path: str):
        """ load a Meter from `path`

        :param path: str of the pickle file to recover Meter from
        :return: a Meter instance
        """
        meter = Meter()
        meter.load_state_dict(load_from_pickle(path))
        return meter
