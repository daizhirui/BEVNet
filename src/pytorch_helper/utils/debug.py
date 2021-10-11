import os


def is_debug():
    return 'DEBUG' in os.environ and int(os.environ['DEBUG'])


def set_debug(debug: bool = True):
    os.environ['DEBUG'] = '1' if debug else '0'


def get_debug_size():
    return int(os.environ.get('DEBUG_SIZE', 32))


def set_debug_size(size: int):
    os.environ['DEBUG_SIZE'] = str(size)
