from .test import Tester


class Evaluator(Tester):
    def __init__(self, arg_cls, register_func):
        super(Evaluator, self).__init__(arg_cls, register_func, 'eval')
