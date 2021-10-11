import os


def make_dirs(path: str):
    """ create directories of `path`

    :param path: str of the directory path to create
    """
    os.makedirs(os.path.abspath(path), exist_ok=True)


def make_dirs_for_file(path: str):
    """ create the folder for the file specified by `path`

    :param path: str of the file
    """
    dir_path = os.path.dirname(path)
    if len(dir_path) == 0:
        dir_path = os.curdir
    os.makedirs(dir_path, exist_ok=True)
