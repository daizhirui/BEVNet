"""
This file contains functions for waiting gpus to be ready when there are other
processes using the requested gpus.
"""
import sys
import time

import torch

from .gpustat import GPUStatCollection
from ..dist import synchronize
from ..log import get_logger

__all__ = [
    'collect_cuda_device',
    'wait_gpus'
]
logger = get_logger(__name__)


def collect_cuda_device(cuda_id: int, mb_size: int = None) -> torch.Tensor:
    """ collect the memory of a gpu

    :param cuda_id: CUDA device index
    :param mb_size: int of memory size in MB to collect
    :return: torch.Tensor of the requested memory
    """
    logger.info(f'Collect CUDA {cuda_id}')
    torch.cuda.set_device(cuda_id)
    if mb_size is None:
        q = GPUStatCollection.new_query()[cuda_id]
        mb_size = q.memory_total
    MB = 1024 * 1024
    size = int(0.80 * mb_size * MB / 4)
    block = torch.empty(size, dtype=torch.float32).cuda()
    return block


def wait_gpus(gpus: dict, collect=True, pre_release=True, sync=True):
    """ wait requested gpus' become available (memory becomes empty)

    :param gpus: dict of gpu mapping: cuda index -> gpu index
    :param collect: Bool to collect the gpu when its memory becomes empty
    :param pre_release: Bool to release all the collected GPU memory before
        returning from this function
    :param sync: Bool to synchronize all the processes such that they release
        the gpu memory together
    """
    if gpus is None:
        logger.warn('`gpus` is None, potential bug in gpu management module')
        return
    gpus_not_ready = True
    blocks = dict()
    collected = dict()
    while gpus_not_ready:
        time.sleep(5)
        gpus_not_ready = False
        queries = GPUStatCollection.new_query()

        for cuda_id, gpu_id in gpus.items():
            if collected.get(gpu_id, False):
                continue
            q = queries[gpu_id]
            if q.processes is not None and len(q.processes) > 0:
                if sys.platform == 'win32':
                    # on windows, GPU memory are managed by WDDM subsystems
                    # nvidia-smi cannot get GPU memory usage of each process
                    # so, all the detected python.exe will be considered
                    processes = [
                        p for p in q.processes
                        if p['command'].lower() == 'python.exe'
                    ]
                else:
                    processes = [
                        p for p in q.processes
                        if p['gpu_memory_usage'] is not None
                    ]
                    processes = [
                        p for p in processes
                        if p['gpu_memory_usage'] > 0
                    ]
                if len(processes) > 0:
                    logger.info(
                        f'GPU {gpu_id} is used, check again in 5 seconds'
                    )
                    gpus_not_ready = True
                    break
            else:
                if q.processes is None:
                    logger.warn(f'GPU {gpu_id} processes is None')
                logger.info(f'GPU {gpu_id} is ready')
                if collect:
                    blocks[gpu_id] = collect_cuda_device(
                        cuda_id, q.memory_total
                    )

    logger.info(f'GPUs {gpus.values()} are ready!')

    if sync:
        synchronize()

    del blocks
    if pre_release:
        torch.cuda.empty_cache()
