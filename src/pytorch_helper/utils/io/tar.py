import os.path
import tarfile
from typing import Callable

from ..log import get_logger

__all__ = [
    'make_tar_file'
]
logger = get_logger(__name__)


def make_tar_file(src: str, dst: str, include: Callable = None):
    """ make a tar file from `src` to `dst`

    :param src: str of the path of source
    :param dst: str of the path to save the tar file
    :param include: Callable to determine which files to include in the tar file
    """
    with tarfile.open(dst, 'w:gz') as file:
        file.add(src, arcname=os.path.basename(src), filter=include)
    logger.info(f'Save {dst}')
