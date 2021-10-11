from dataclasses import asdict
from dataclasses import dataclass
from typing import Type
from typing import TypeVar

import ruamel.yaml as yaml

from ...utils.io import load_yaml
from ...utils.io import save_yaml
from ...utils.log import get_logger

T = TypeVar('T')

__all__ = ['OptionBase']

logger = get_logger(__name__)


class _Base:

    # implement class methods
    @classmethod
    def from_dict(cls: Type[T], option_dict: dict, **kwargs) -> T:
        """ Build option from a dict of arguments

        :param option_dict: Dict of arguments
        :param kwargs: extra keyword arguments
        :return: T instance
        """
        if option_dict is None:
            return None
        option_dict.update(kwargs)
        logger.info(f'create {cls.__name__} from dict')
        return cls(**option_dict)

    @classmethod
    def from_file(cls: Type[T], option_file: str, **kwargs) -> T:
        """ Build option from a yaml file

        :param option_file: Path to the yaml file
        :param kwargs: extra keyword arguments
        :return: T instance
        """
        if option_file is None:
            return None
        option_dict = load_yaml(option_file)
        option_dict.update(kwargs)
        logger.info(f'create {cls.__name__} from file {option_file}')
        return cls(**option_dict)


@dataclass()
class OptionBase(_Base):

    def as_dict(self) -> dict:
        """ convert self to Dict

        :return: Dict of self's attributes
        """
        return asdict(self)

    def save_as_yaml(self, output_path: str):
        """ save self as a yaml file

        :param output_path: path to output
        :return:
        """
        save_yaml(output_path, self.as_dict())

    def __str__(self):
        """ convert self to a pretty yaml string the same as the content in the
        file created by `self.save_as_yaml`

        :return: str
        """
        yaml_obj = yaml.YAML()
        yaml_obj.indent(mapping=4, sequence=4)

        class MySteam:
            def __init__(self):
                self.s = ''

            def write(self, s):
                self.s += s.decode('utf-8')

        stream = MySteam()
        yaml_obj.dump(self.as_dict(), stream)
        return stream.s
