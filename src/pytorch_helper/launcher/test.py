import os

from .launcher import Launcher

__all__ = ['Tester']


class Tester(Launcher):
    def __init__(self, arg_cls, register_func, mode='test'):
        super(Tester, self).__init__(arg_cls, register_func, mode=mode)

    def modify_task_dict(self, task_dict):
        task_dict = super(Tester, self).modify_task_dict(task_dict)
        task_dict['resume'] = False
        if self.args.pth_path:
            task_dict['model']['pth_path'] = self.args.pth_path

        # dataset path
        if self.args.dataset_path is not None:
            task_dict['dataset_path'] = self.args.dataset_path
        if not os.path.exists(task_dict['dataset_path']):
            raise FileNotFoundError(
                f'{task_dict["dataset_path"]} does not exist.'
            )

        # output path
        if self.args.output_path is not None:
            task_dict['output_path'] = self.args.output_path
        elif not os.path.exists(task_dict['output_path']):
            cur_output_path = os.path.dirname(self.args.task_option_file)
            datetime = os.path.basename(cur_output_path)
            cur_output_path = os.path.dirname(cur_output_path)
            task_name = os.path.basename(cur_output_path)
            cur_output_path = os.path.dirname(cur_output_path)
            task_dict['name'] = task_name
            task_dict['datetime'] = datetime
            task_dict['output_path'] = cur_output_path

        if int(os.environ['DEBUG']):
            task_dict['dataloader']['kwargs']['num_workers'] = 0

        return task_dict
