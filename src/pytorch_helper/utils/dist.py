"""
This file contains functions for gpu communication
"""
from typing import Dict
from typing import Hashable
from typing import Union

import torch
import torch.distributed as pt_dist

from .pre_pytorch_init import ready_for_torch

ready_for_torch(assert_=True)
ReduceOp = getattr(pt_dist, 'ReduceOp')


def is_distributed() -> bool:
    """
    :return: True if torch.distributed is initialized, which means a distributed
        multi-gpu task may be launched
    """
    return pt_dist.is_available() and pt_dist.is_initialized()


def get_world_size() -> int:
    """
    :return: the number of processes in the current process group
    """
    if is_distributed():
        return pt_dist.get_world_size()
    return 1


def get_rank(group=pt_dist.group.WORLD) -> int:
    """
    :param group: the process group to work on, default is group.WORLD
    :return: the rank of the current process in ``group``, 0 to ``world_size``.
        -1 if the current process is not part of ``group``.
    """
    if is_distributed():
        return pt_dist.get_rank(group)
    return 0


def is_rank0() -> bool:
    """ rank 0 process is considered as the main process, you can do some single
    process operations such as disk IO, visualization and etc. on the main
    process.
    :return: True if the current process's rank is 0
    """
    return get_rank() == 0


def synchronize():
    """ put a barrier to make all processes reach the same point.
    """
    if is_distributed():
        if get_world_size() > 1:
            pt_dist.barrier()


def reduce_value(
    input_value: Union[torch.Tensor, Dict[Hashable, torch.Tensor]],
    rank0_only: bool = False, average: bool = True,
    op=ReduceOp.SUM
):
    """ Reduce the values in ``input_dict`` with operation ``op``.

    :param input_value: value or dict of values to reduce
    :param rank0_only: only rank0 process will receive the reduce result
    :param average: average the reduced result by the world size
    :param op: reduce operation, default is ``ReduceOp.SUM``
    :return: reduced value or dict of reduced values
    """
    if input_value is None:
        return input_value
    world_size = get_world_size()
    if world_size < 2:
        return input_value
    reduce_func = pt_dist.reduce if rank0_only else pt_dist.all_reduce
    with torch.no_grad():
        if isinstance(input_value, dict):
            output = dict()
            for k, v in input_value.items():
                output[k] = reduce_value(v, rank0_only, average, op)
        elif isinstance(input_value, (list, tuple)):
            output = [reduce_value(x, rank0_only, average, op)
                      for x in input_value]
        else:
            output = input_value
            reduce_func(input_value, op=op)
            if average:
                if rank0_only:
                    if is_rank0():
                        output /= world_size
                else:
                    output /= world_size
    return output
