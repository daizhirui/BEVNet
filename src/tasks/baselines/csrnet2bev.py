from tasks.baselines.dssinet2bev import DSSINet2BEVTask
from pytorch_helper.settings.spaces import Spaces


@Spaces.register(Spaces.NAME.TASK, 'CSRNet2BEVTask')
class CSRNet2BEVTestTask(DSSINet2BEVTask):
    pass
