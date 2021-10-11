from .launcher import Launcher

__all__ = ['Trainer']


class Trainer(Launcher):
    def __init__(self, arg_cls, register_func):
        super(Trainer, self).__init__(arg_cls, register_func, mode='train')
