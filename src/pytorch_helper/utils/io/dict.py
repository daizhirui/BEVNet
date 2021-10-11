import csv
import os.path
from collections import OrderedDict

from .make_dirs import make_dirs_for_file
from .yaml import save_yaml

__all__ = [
    'save_dict_as_csv',
    'save_dict_as_yaml'
]


def save_dict_as_csv(path: str, a: OrderedDict, append: bool = False):
    """ save dict `a` as a csv file to `path`

    :param a: OrderedDict to save
    :param path: str of the csv file path
    :param append: Bool to append `a` as a row to the file at `path`
    """
    make_dirs_for_file(path)

    new_csv = True
    fieldnames = list(a.keys())
    if os.path.exists(path) and append:
        new_csv = False
        with open(path, 'r') as file:
            old_fieldnames = file.readline().strip().split(',')
            for fieldname in fieldnames:
                if fieldname not in old_fieldnames:
                    old_fieldnames.append(fieldname)
                    new_csv = True
            fieldnames = old_fieldnames
    else:
        fieldnames = list(a.keys())

    with open(path, 'a+' if append else 'w', newline='') as file:
        csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
        if new_csv:
            csv_writer.writeheader()
        for k in fieldnames:
            if k not in a:
                a[k] = 'None'
        csv_writer.writerow(a)


def save_dict_as_yaml(path, a: dict):
    """ save dict as a yaml file

    :param path: str of the file path
    :param a: dict to save
    """
    save_yaml(path, a)
