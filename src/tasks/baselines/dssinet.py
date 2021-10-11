from pytorch_helper.settings.options import TaskOption
from pytorch_helper.settings.spaces import Spaces
from tasks.ivnet import IVNetTask

__all__ = ['DSSINetTask']


@Spaces.register(Spaces.NAME.TASK, 'DSSINetTask')
class DSSINetTask(IVNetTask):

    def __init__(self, task_option: TaskOption):
        super(DSSINetTask, self).__init__(task_option)
        self.iv_map_key = 'head_map'
