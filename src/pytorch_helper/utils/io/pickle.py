import pickle
from typing import Any

from . import config
from .make_dirs import make_dirs_for_file
from ..log import get_logger

__all__ = [
    'save_as_pickle',
    'load_from_pickle'
]

logger = get_logger(__name__)


def save_as_pickle(path: str, result: Any):
    """ save `result` as a pickle file to `path`

    :param path: str of the file path
    :param result: content to save
    """
    make_dirs_for_file(path)
    with open(path, 'wb') as f:
        pickle.dump(result, f)
    if not config.silent:
        logger.info(f'Save {path}')


def load_from_pickle(path: str):
    """ load data from pickle file

    :param path: str of the path of the pickle file
    """
    if not config.silent:
        logger.info(f'Load from {path}')
    with open(path, 'rb') as f:
        return pickle.load(f)
