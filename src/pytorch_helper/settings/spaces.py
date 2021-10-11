from collections import Iterable
from collections import defaultdict
from enum import Enum

from ..utils.log import get_logger

logger = get_logger(__name__)


class Spaces:
    _registry = defaultdict(dict)

    class NAME(Enum):
        MODEL = 'model'
        DATALOADER = 'dataloader'
        LOSS_FN = 'loss_fn'
        METRIC = 'metric'
        TASK_OPTION = 'task_option'
        TASK = 'task_for_train'
        TASK_FOR_TEST = 'task_for_test'

    def __init_subclass__(cls, **kwargs):
        space = kwargs.pop('space')
        space = Spaces.NAME(space)
        ref = kwargs.pop('ref')
        super().__init_subclass__(**kwargs)

        cls._registry[space][ref] = cls

    @classmethod
    def add_space(cls, name, value):
        a = {x.name: x.value for x in cls.NAME}
        a[name] = value
        cls.NAME = Enum('NAME', a)

    @staticmethod
    def build(space, ref, kwargs):
        space = Spaces.NAME(space)
        logger.info(f'build in {space} Space with reference: {ref}')
        return Spaces._registry[space][ref](**kwargs)

    @staticmethod
    def register(spaces, refs):

        if isinstance(refs, str):
            refs = [refs]

        if isinstance(spaces, Spaces.NAME):
            spaces = [spaces] * len(refs)
        elif not isinstance(spaces, Iterable):
            raise ValueError(f'{spaces} should be a {Spaces.NAME} or a list '
                             f'of {Spaces.NAME}.')

        def decorator(cls):
            for space, ref in zip(spaces, refs):
                space = Spaces.NAME(space)
                Spaces._registry[space][ref] = cls
                logger.info(
                    f'register {cls} in {space} Space with reference {ref}'
                )

            return cls

        return decorator
