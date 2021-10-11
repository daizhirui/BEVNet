import sys
from datetime import datetime
from logging import CRITICAL
from logging import DEBUG
from logging import ERROR
from logging import INFO
from logging import NOTSET
from logging import WARNING
from typing import Iterable

import colorama
import colorlog
from ruamel import yaml
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

from .pre_pytorch_init import get_cuda_visible_devices
from .pre_pytorch_init import ready_for_torch

colorama.init()

bar = tqdm
bar_len = 80
verbose_level = DEBUG

__all__ = [
    'notebook_compatible',
    'pbar',
    'get_logger',
    'NOTSET',
    'DEBUG',
    'INFO',
    'WARNING',
    'ERROR',
    'CRITICAL',
    'pretty_dict',
    'get_datetime'
]


def notebook_compatible():
    """ setup the module to make it compatible with jupyter notebook
    """
    global bar
    bar = tqdm_notebook
    global bar_len
    bar_len = None


def pbar(iterable: Iterable = None, ncols: int = bar_len, **kwargs):
    """ create a tqdm bar

    :param iterable: iterable object
    :param ncols: int of progress bar length
    :param kwargs: extra keyword arguments to create the progress bar
    :return:
    """
    return bar(iterable, ncols=ncols, **kwargs)


class TqdmStream:
    @staticmethod
    def write(msg):
        tqdm.write(msg, end='')

    @staticmethod
    def flush():
        return


class ExitHandler(colorlog.StreamHandler):
    def emit(self, record):
        if record.levelno >= ERROR:
            super(ExitHandler, self).emit(record)
            sys.exit(1)


exit_on_error = False
loggers = {}


def get_logger(name: str):
    global loggers
    if name in loggers:
        return loggers[name]

    fmt = f'%(log_color)s[{_get_device()}][%(process)d][%(asctime)s]' \
          f'[%(levelname)s: %(name)s: %(lineno)4d]: %(message)s'
    handler = colorlog.StreamHandler(stream=TqdmStream)
    handler.setFormatter(colorlog.ColoredFormatter(fmt))

    logger = colorlog.getLogger(name)
    logger.addHandler(handler)
    logger.setLevel(verbose_level)
    if exit_on_error:
        handler = ExitHandler(stream=TqdmStream)
        handler.setFormatter(colorlog.ColoredFormatter(fmt))
        logger.addHandler(handler)
    logger.propagate = False
    loggers[name] = logger
    return logger


def pretty_dict(a: dict) -> str:
    """ convert dict `a` to str in pretty yaml format

    :param a: dict to convert
    :return: str
    """
    yaml_obj = yaml.YAML()
    yaml_obj.indent(mapping=4, sequence=4)

    class MySteam:
        def __init__(self):
            self.s = ''

        def write(self, s):
            self.s += s.decode('utf-8')

    stream = MySteam()
    yaml_obj.dump(a, stream)
    return stream.s


def _get_device() -> str:
    """ get the str of current GPU device
    """
    if not ready_for_torch():
        # should not import torch yet
        return 'CPU'

    import torch.cuda
    import torch.distributed as pt_dist
    from .dist import is_distributed

    device = ''
    try:
        visible_devices = get_cuda_visible_devices()
        if is_distributed():
            rank = pt_dist.get_rank()
            gpu_id = visible_devices[rank]
            device = f'RANK{rank} on GPU{gpu_id}'
        elif torch.cuda.is_available():
            gpu_id = visible_devices[torch.cuda.current_device()]
            device = f'GPU{gpu_id}'
    except KeyError:
        if torch.cuda.is_available():
            device = f'GPU{torch.cuda.current_device()}'
        else:
            device = 'CPU'
    return device


def get_datetime() -> str:
    """ get the str of current date and time
    """
    return datetime.now().strftime('%b%d_%H-%M-%S')
