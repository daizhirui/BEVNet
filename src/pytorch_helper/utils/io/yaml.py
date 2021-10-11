import os

from ruamel import yaml

from .make_dirs import make_dirs_for_file
from ..log import get_logger

__all__ = [
    'load_yaml',
    'merge_task_dicts',
    'load_task_option_yaml',
    'save_yaml'
]

logger = get_logger(__name__)


def load_yaml(
    path: str,
    recursive: bool = True,
    recursive_mark: str = '<<'
) -> dict:
    """ load dict from yaml file

    :param path: str of the path of the yaml file
    :param recursive:
    :param recursive_mark:
    :return: dict of the yaml file
    """
    logger.info(f'Load from {path}')
    with open(path, 'r') as file:
        a = yaml.safe_load(file)

    if not recursive:
        return a

    def _check_dict(d: dict):
        for k, v in d.items():
            if isinstance(v, str):
                if v.startswith(recursive_mark):
                    v = v[2:].strip()
                    v_path = os.path.join(os.path.dirname(path), v)
                    logger.info(f'Load key {k} from {v} recursively')
                    d[k] = load_yaml(v_path, recursive, recursive_mark)
            elif isinstance(v, dict):
                d[k] = _check_dict(v)
            elif isinstance(v, list):
                d[k] = _check_list(v)
        return d

    def _check_list(d):
        for i in range(len(d)):
            v = d[i]
            if isinstance(v, str):
                if v.startswith(recursive_mark):
                    v = v[2:].strip()
                    v_path = os.path.join(os.path.dirname(path), v)
                    logger.info(f'Load list item{i} {v} recursively')
                    d[i] = load_yaml(v_path, recursive, recursive_mark)
            elif isinstance(v, dict):
                d[i] = _check_dict(v)
            elif isinstance(v, list):
                d[i] = _check_list(v)
        return d

    check_func = {
        _check_dict.__name__: _check_dict,
        _check_list.__name__: _check_list
    }

    a = check_func.get(f'_check_{type(a).__name__}', lambda x: x)(a)

    return a


def merge_task_dicts(base: dict, update: dict):
    if base is None:
        return update

    if update is None:
        return base

    out = base.copy()

    for key, value in update.items():
        assert key in base, f'key {key} not found in base dict'
        if isinstance(value, dict):
            out[key] = merge_task_dicts(base[key], value)
        else:
            out[key] = value
    return out


def load_task_option_yaml(path: str):
    task_dict = load_yaml(path)
    if 'base' in task_dict:
        base_task_option_file = task_dict.pop('base')
        base_task_option_file = os.path.join(
            os.path.dirname(path), base_task_option_file
        )
        base_task_dict = load_task_option_yaml(base_task_option_file)
        try:
            task_dict = merge_task_dicts(base_task_dict, task_dict)
        except AssertionError as e:
            logger.error(f'When loading {path}, the following error occurs:')
            raise e
    return task_dict


def save_yaml(path: str, a: dict):
    """ save dict as a yaml file

    :param path: str of the file path
    :param a: dict to save
    """
    make_dirs_for_file(path)

    yaml_obj = yaml.YAML()
    yaml_obj.indent(mapping=4, sequence=6, offset=4)
    with open(path, 'w') as file:
        yaml_obj.dump(a, file)
    logger.info(f'Save {path}')
