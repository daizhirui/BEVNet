import time

import torch

from . import config
from .make_dirs import make_dirs_for_file
from ..log import get_logger

__all__ = [
    'load_pth',
    'save_pth'
]

logger = get_logger(__name__)


def load_pth(path: str) -> dict:
    """ load state dict from a checkpoint file

    :param path: str of path of the checkpoint file
    :return: dict
    """
    if not config.silent:
        logger.info(f'Load state dict from {path}')
    t = time.time()
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    t = time.time() - t
    if not config.silent:
        logger.info(f'Loaded, took {t:.2f} seconds')
    return state_dict


def save_pth(path: str, state_dict: dict):
    """ save `state_dict` as a checkpoint file to `path`

    :param path: str of the path to save the checkpoint
    :param state_dict: dict
    """
    make_dirs_for_file(path)
    if not config.silent:
        logger.info(f'Save state dict to {path}')
    t = time.time()
    if torch.__version__ >= '1.6.0':
        torch.save(state_dict, path, _use_new_zipfile_serialization=False)
    else:
        torch.save(state_dict, path)
    t = time.time() - t
    if not config.silent:
        logger.info(f'Saved, took {t:.2f} seconds')
