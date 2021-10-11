__all__ = ['register_func']


def register_func():
    import torch.nn.modules as nn
    from pytorch_helper.settings.spaces import Spaces
    from pytorch_helper.settings.options import TaskOption

    import tasks

    tasks

    Spaces.register(
        Spaces.NAME.TASK_OPTION,
        ['IVNetTask', 'PoseNetTask']
    )(TaskOption)

    Spaces.register(Spaces.NAME.LOSS_FN, nn.MSELoss.__name__)(nn.MSELoss)
    # print('called')
    # from tasks.bevnet import BEVNetTask
    # from models.bevnet import BEVNet

# def register_func():
#     import torch.nn.modules as nn
#     from pytorch_helper.settings.options import TaskOption
#     from pytorch_helper.settings.spaces import Spaces
#
#     import datasets
#     import models
#     import tasks
#     from settings import options
#
#     # task for train
#     Spaces.register_task_for_train(tasks.BEVNetTrainTask, 'BEVNetTask')
#     Spaces.register_task_for_train(tasks.IVNetTrainTask, 'IVNetTask')
#     Spaces.register_task_for_train(tasks.PoseNetTrainTask, 'PoseNetTask')
#     Spaces.register_task_for_train(tasks.DSSINetTrainTask, 'DSSINetTask')
#     Spaces.register_task_for_train(tasks.CSRNetTrainTask, 'CSRNetTask')
#     Spaces.register_task_for_train(tasks.RCNNTrainTask, 'RCNNTask')
#     # task for test
#     Spaces.register_task_for_test(tasks.BEVNetTestTask, 'BEVNetTask')
#     Spaces.register_task_for_test(tasks.IVNetTestTask, 'IVNetTask')
#     Spaces.register_task_for_test(tasks.IV2BEVTestTask, 'IV2BEVTask')
#     Spaces.register_task_for_test(tasks.PoseNetTestTask, 'PoseNetTask')
#     Spaces.register_task_for_test(tasks.DSSINetTestTask, 'DSSINetTask')
#     Spaces.register_task_for_test(tasks.DSSINet2BEVTestTask, 'DSSINet2BEVTask')
#     Spaces.register_task_for_test(tasks.CSRNetTestTask, 'CSRNetTask')
#     Spaces.register_task_for_test(tasks.CSRNet2BEVTestTask, 'CSRNet2BEVTask')
#     Spaces.register_task_for_test(tasks.RCNN2BEVTestTask, 'RCNN2BEVTask')
#     Spaces.register_task_for_test(tasks.CSPNet2BEVTestTask, 'CSPNet2BEVTask')
#     # task option
#     Spaces.register_task_option(options.BEVNetTaskOption, 'BEVNetTask')
#     Spaces.register_task_option(options.BEVNetTaskOption, 'IV2BEVTask')
#     Spaces.register_task_option(TaskOption, 'IVNetTask')
#     Spaces.register_task_option(TaskOption, 'PoseNetTask')
#     Spaces.register_task_option(options.DSSINetTaskOption, 'DSSINetTask')
#     Spaces.register_task_option(options.DSSINetTaskOption, 'DSSINet2BEVTask')
#     Spaces.register_task_option(options.CSRNetTaskOption, 'CSRNetTask')
#     Spaces.register_task_option(options.CSRNetTaskOption, 'CSRNet2BEVTask')
#     Spaces.register_task_option(options.RCNNTaskOption, 'RCNNTask')
#     Spaces.register_task_option(options.RCNNTaskOption, 'RCNN2BEVTask')
#     Spaces.register_task_option(options.CSPNetTaskOption, 'CSPNet2BEVTask')
#     # model
#     Spaces.register_model(models.IVNet)
#     Spaces.register_model(models.PoseNet)
#     Spaces.register_model(models.BEVNet)
#     Spaces.register_model(models.BEVNetFeetOnly)
#     Spaces.register_model(models.BEVNetNoAttention)
#     # metrics
#     Spaces.register_metric(models.DensityCluster)
#     Spaces.register_metric(models.IndividualDistance)
#     # loss
#     Spaces.register_loss_fn(models.BEVLoss)
#     Spaces.register_loss_fn(models.MagnifiedMSELoss)
#     Spaces.register_loss_fn(models.PoseLoss)
#     Spaces.register_loss_fn(nn.MSELoss)
#     # dataset
#     Spaces.register_dataloader(datasets.CityUHKBEVLoaders)
