import os
import random
from abc import ABC
from collections import defaultdict
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from .launcher.launcher import LauncherTask
from .settings.options import TrainRoutine
from .utils.dist import synchronize
from .utils.io import save_dict_as_csv
from .utils.io import save_pth
from .utils.log import get_logger
from .utils.log import pbar
from .utils.meter import Meter

__all__ = ['Task', 'Batch']

from .utils.timer import TimerManager

logger = get_logger(__name__)


@dataclass
class Batch:
    gt: Any
    size: int = None
    pred: Any = None
    loss: Any = None

    @property
    def batch(self):
        return self.gt


class Task(LauncherTask, ABC):

    class STAGE(Enum):
        TRAIN = 'train'
        VALID = 'valid'
        TEST = 'test'
        ALL = 'all'

    def __init__(self, task_option):
        self.init_random_seed()

        self._option = task_option
        self.timer_manager = TimerManager()

        logger.info(f'Task: {self.option.name}')
        logger.info(f'Datetime: {self.option.datetime}')

        from .utils.dist import get_rank
        self.rank = get_rank()
        self.is_rank0 = self.rank == 0
        logger.info(f'I am rank{self.rank}.')

        # setup device
        torch.cuda.set_device(self._option.cuda_ids[0])
        logger.info(f'Pick CUDA:{self._option.cuda_ids[0]}.')

        if self._option.distributed:  # DistributedDataParallel (DDP)
            logger.info('Adjust batch size per process for DDP.')
            from .utils.dist import get_world_size
            n_gpus = get_world_size()
            batch_size = self.option.dataloader.kwargs['batch_size']
            self.option.dataloader.kwargs['batch_size'] //= n_gpus
            self.option.dataloader.kwargs['num_workers'] //= n_gpus
            rest = batch_size % n_gpus
            if rest > 0:
                raise RuntimeError(
                    f'Unbalanced batch size distribution: batch size '
                    f'is {self.option.dataloader.kwargs["batch_size"]} '
                    f'for {n_gpus} GPUs.'
                )

        # load model
        logger.info('Loading model ...')
        with self.timer_manager.timing() as timer:
            self.model, state_dict = self.option.model.build()
        logger.info(f'Loaded, took {timer.elapsed} seconds.')
        logger.info(f'Moving model to CUDA ...')
        with self.timer_manager.timing() as timer:
            self.model.cuda()
        logger.info(f'Moved, took {timer.elapsed} seconds.')

        if self._option.distributed:
            logger.info('Use DDP, convert BatchNorm to SyncBatchNorm.')
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

            from torch.nn.parallel import DistributedDataParallel

            self.model = DistributedDataParallel(
                module=self.model,
                device_ids=self._option.cuda_ids,
                output_device=self._option.cuda_ids[0],
                # allow model to be partially updated
                find_unused_parameters=True
            )
            self.unwrapped_model = self.model.module
        elif len(self._option.cuda_ids) > 1:
            logger.info('Use DataParallel.')

            from torch.nn.parallel import DataParallel

            self.model = DataParallel(
                module=self.model,
                device_ids=self._option.cuda_ids,
                output_device=self._option.cuda_ids[0]
            )
            self.unwrapped_model = self.model.module
        else:
            self.unwrapped_model = self.model

        # loss function
        self.loss_fn = self.option.loss.build()
        self.loss_min = None
        self.load_state(self.loss_fn, state_dict, 'loss_fn')
        # dataloader
        self.dataloader = self.option.dataloader.build()
        self.cur_dataloader = None
        # optimizer
        self.optimizer = None
        self.lr_scheduler = None
        if self._option.train:
            self.optimizer = self.option.optimizer.build(self.unwrapped_model)
            self.lr_scheduler = self.option.lr_scheduler.build(self.optimizer)
        # tracking
        self.epoch = self.option.train_setting.start_epoch
        self.current_train_routine = None
        if self._option.train:
            self.cur_stage = self.STAGE.TRAIN
        else:
            self.cur_stage = self.STAGE.TEST
        self.batch_cnt = {
            self.STAGE.TRAIN: 0,
            self.STAGE.VALID: 0,
            self.STAGE.TEST: 0,
            self.STAGE.ALL: 0
        }
        # logging
        self.tboard = None
        self.in_stage_meter_keys = set()
        self.model_output_dict = defaultdict(list)
        self.keep_model_output = False
        self.progress_bars = None
        if self.is_rank0:
            if self._option.train:
                self.progress_bars = {
                    self.STAGE.ALL: pbar(
                        initial=self.epoch,
                        total=self.option.train_setting.epochs,
                        position=0, desc='Epoch'
                    ),
                    self.STAGE.TRAIN: pbar(
                        position=1, desc='Train'
                    ),
                    self.STAGE.VALID: pbar(
                        position=2, desc='Valid'
                    ),
                    self.STAGE.TEST: pbar(
                        position=3, desc=' Test'
                    )
                }
                logger.info("Initialize tensorboard")
                self.tboard = SummaryWriter(log_dir=self.option.output_path_tb)
            else:
                self.keep_model_output = True
                self.progress_bars = {
                    self.STAGE.TEST: pbar(position=0, desc=' Test')
                }

        # logging
        path = f'meter-{"train" if self.option.train else "test"}.pkl'
        path = os.path.join(self.option.output_path_tb, path)
        if os.path.exists(path) and self.option.resume:
            self.meter = Meter.load(path)
        else:
            self.meter = Meter()
        self.in_stage_logged = False

        # resume
        self.resume(state_dict)

    @property
    def option(self):
        return self._option

    def resume(self, state_dict):
        if state_dict is None:
            return

        if self._option.train and not self._option.resume:
            return

        key = self.StateKey.EPOCH.value
        if key in state_dict:
            self.epoch = state_dict[key] + 1
            logger.info(f"Resume from epoch {state_dict[key]}")

        if self._option.train:
            key = self.StateKey.OPTIMIZER.value
            self.load_state(self.optimizer, state_dict, key)
            key = self.StateKey.LR_SCHEDULER.value
            self.load_state(self.lr_scheduler, state_dict, key)
            key = self.StateKey.RNG_STATE.value
            if key in state_dict:
                self.set_rng_state(state_dict[key])
            else:
                logger.warn(f'No random state to resume!')

            # miscellaneous
            key = self.StateKey.LOSS_MIN.value
            self.loss_min = state_dict.get(key, self.loss_min)
            key = self.StateKey.BATCH_CNT.value
            self.batch_cnt = state_dict.get(key, self.batch_cnt)
            key = self.StateKey.IN_STAGE_METER_KEYS.value
            self.in_stage_meter_keys = state_dict.get(
                key, self.in_stage_meter_keys
            )
            if self.is_rank0:
                self.progress_bars[self.STAGE.ALL].update(self.epoch)

    ################
    # RANDOM STATE #
    ################

    @staticmethod
    def init_random_seed(seed: int = 0):
        """ set the initial random seed of torch, torch.cuda, numpy, and random

        :param seed: int
        """
        logger.info(f'Init random seeds to {seed}')
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    @staticmethod
    def set_rng_state(rng_state: dict):
        """ set the states of all the random generators

        :param rng_state: dict of random generator states
        """
        set_state_fns = dict(
            numpy=np.random.set_state,
            random=random.setstate,
            torch=torch.set_rng_state,
            torch_cuda=torch.cuda.set_rng_state_all
        )
        for key, set_state_fn in set_state_fns.items():
            if key in rng_state:
                set_state_fn(rng_state[key])
            else:
                logger.warn(f'random state for {key} is missing!')

    @staticmethod
    def get_rng_state():
        """ get the states of all the random generators

        :return: dict of random generator states
        """
        seed = dict(
            numpy=np.random.get_state(),
            random=random.getstate(),
            torch=torch.get_rng_state(),
            torch_cuda=torch.cuda.get_rng_state_all()
        )
        return seed

    ##############
    # TASK STATE #
    ##############

    class StateKey(Enum):
        OPTION = 'option'
        MODEL = 'model'
        LOSS_FN = 'loss_fn'
        EPOCH = 'epoch'
        RNG_STATE = 'rng_state'
        LR = 'lr'
        OPTIMIZER = 'optimizer'
        LR_SCHEDULER = 'lr_scheduler'
        LOSS_MIN = 'loss_min'
        IN_STAGE_METER_KEYS = 'in_stage_meter_keys'
        BATCH_CNT = 'batch_cnt'

    def state_dict(self, resumable):

        def get_state(obj):
            if obj and hasattr(obj, 'state_dict'):
                return obj.state_dict()

        state_dict = {
            self.StateKey.OPTION.value: self.option.as_dict(),
            self.StateKey.MODEL.value: get_state(self.unwrapped_model),
            self.StateKey.LOSS_FN.value: get_state(self.loss_fn),
            self.StateKey.EPOCH.value: self.epoch,
            self.StateKey.RNG_STATE.value: self.get_rng_state()
        }

        if not resumable:
            return state_dict

        state_dict.update({
            self.StateKey.LR.value: self.learning_rate,
            self.StateKey.OPTIMIZER.value: get_state(self.optimizer),
            self.StateKey.LR_SCHEDULER.value: get_state(self.lr_scheduler),
            self.StateKey.LOSS_MIN.value: self.loss_min,
            self.StateKey.IN_STAGE_METER_KEYS.value: self.in_stage_meter_keys,
            self.StateKey.BATCH_CNT.value: self.batch_cnt
        })
        return state_dict

    @staticmethod
    def load_state(obj, state_dict: dict, key: str):
        """ load the state of `obj` from `state_dict[key]`

        :param state_dict: dict of object stats
        :param key: reference to find the state for `obj`
        :param obj: the object to load state
        """
        if state_dict is None or obj is None:
            return
        if key in state_dict:
            state_dict = state_dict[key]
            if state_dict is None:
                return
            if len(state_dict) > 0:
                try:
                    obj.load_state_dict(state_dict)
                except Exception as e:
                    logger.warn(repr(e))

    def save_pth(self, name: str = None, resumable: bool = True):
        """ save the task state as a checkpoint

        :param name: str of the checkpoint file name, default is `epoch_x` where
            x is the current epoch number
        :param resumable: Bool to determine whether to store extra states such
            that a training process can resume from the checkpoint file.
        """
        state_dict = self.state_dict(resumable)

        pth_name = f'{name}.pth' if name else f'epoch_{self.epoch}.pth'
        path = os.path.join(self.option.output_path_pth, pth_name)
        save_pth(path, state_dict)
        logger.info(f"Saved checkpoint: {path} at epoch {self.epoch}")

    def save_best_model(self, valid_loss):
        """ save the model which has minimum validation loss

        :param valid_loss: latest validation loss
        """
        if isinstance(valid_loss, dict):
            if self.loss_min is None:
                self.loss_min = dict()
            for k, v in valid_loss.items():
                v_min = self.loss_min.get(k, None)
                if v_min is None or v_min > v:
                    self.loss_min[k] = v
                    self.save_pth(f'best-{k.replace("/", "-")}')
        else:
            if self.loss_min is None or self.loss_min > valid_loss:
                self.loss_min = valid_loss
                self.save_pth('best')

    def backup(self, immediate: bool = False, resumable: bool = True):
        """ determine if it is time to backup the task and save the task state
        if necessary

        :param immediate: Bool to ignore all the restriction, save the task
            state and the meter state
        :param resumable: Bool to set the checkpoint file resumable
        """
        freq = self.option.train_setting.save_model_freq
        if immediate or (self.epoch > 0 and self.epoch % freq == 0):
            self.save_pth(resumable=resumable)
            path = f'meter-{"train" if self.option.train else "test"}.pkl'
            path = os.path.join(self.option.output_path_tb, path)
            self.meter.save(path)

    ############
    # RUN TASK #
    ############

    # ENTRY

    def run(self):
        """ the entry to launch the task, run the task until `self.epoch` is
        equal to `self.option.training_setting.epochs`, and save the final
        model state as a non-resumable checkpoint file.
        """

        self.profiling()
        getattr(self, f'run_{self._option.task_mode.value}')()

    def run_train(self):
        for epoch in range(self.epoch, self.option.train_setting.epochs):
            if self.is_rank0 and self.tboard is not None:
                lr = self.optimizer.param_groups[0]['lr']
                self.tboard.add_scalar('learning-rate', lr, self.epoch)
            # NOTE: different stage should not share the same set of keys
            self.meter.reset_tags(self.in_stage_meter_keys)
            self.one_epoch(epoch)
        # save only the state dicts of model and loss_fn
        self.save_pth('model_final', resumable=False)

    def run_test(self):
        self.cur_stage = self.STAGE.TEST
        self.current_train_routine = TrainRoutine(epochs=self.epoch)
        self.setup_before_stage()

        self._test()

        summary = OrderedDict()
        summary['name'] = self.option.name
        summary['datetime'] = self.option.datetime
        summary['epoch'] = self.epoch
        if self.option.model.pth_path is None:
            summary['pth_file'] = 'None'
        else:
            summary['pth_file'] = os.path.basename(
                self.option.model.pth_path)
        summary.update(self.summarize_logging_after_stage())

        if self.is_rank0:
            path = os.path.join(self._option.output_path_test,
                                'test-summary.csv')
            save_dict_as_csv(path, summary)
        synchronize()

        for key, value in self.model_output_dict.items():
            self.model_output_dict[key] = np.concatenate(value, axis=0)

        return summary

    def run_eval(self):
        logger.warn(f'{type(self).__name__} does not implement run_eval, this '
                    f'is expected to be implemented by the user. Do not call '
                    f'this in your implementation of run_eval to avoid this '
                    f'warning.')

    # PROFILING

    def profiling(self):
        if not self.option.profiling:
            return
        if self.option.train:
            self.cur_stage = self.STAGE.TRAIN
            batch_func = self.train
        else:
            self.cur_stage = self.STAGE.TEST
            batch_func = self.test
        self.setup_before_stage()

        logger.info(f'Start {self.option.profile_tool}')

        if self.option.profile_tool == 'cprofile':
            import cProfile

            with cProfile.Profile() as profiler:
                with torch.enable_grad():
                    for batch in self.load_batch():
                        batch_func(batch)
                        synchronize()
                        self.update_logging_in_stage(batch)
                        synchronize()

                path = os.path.join(
                    self.option.output_path_tb, 'profile_result.cprofile'
                )
                profiler.dump_stats(path)

            logger.info('Visualize profiling result')
            try:
                os.system(f'snakeviz {path}')
            except KeyboardInterrupt:
                exit(0)

        elif self.option.profile_tool == 'torch':
            from .utils.profiler import PatchedProfiler

            with PatchedProfiler(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA],
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    self.option.output_path_tb
                ),
                profile_memory=self.option.profile_memory,
                with_stack=True
            ) as profiler:
                with torch.enable_grad():
                    for batch in self.load_batch():
                        batch_func(batch)
                        synchronize()
                        self.update_logging_in_stage(batch)
                        synchronize()
                        profiler.step()

            logger.info('Visualize profiling result')

            try:
                os.system(f'tensorboard '
                          f'--logdir {self.option.output_path_tb} '
                          f'--port 6005 ')
            except KeyboardInterrupt:
                exit(0)
        else:
            raise ValueError(f'Unknown profiling tool: '
                             f'{self.option.profile_tool}')

        exit(0)

    # EPOCH ENTRY

    def one_epoch(self, epoch: int):
        """ run a epoch of the task, including training, validation and test if
        `self.option.train_setting.valid_on_test` is positive integer.

        :param epoch: int of current epoch number
        :return: dict of validation result
        """
        self.epoch = epoch
        self.dataloader.set_epoch(epoch)

        self._train()
        self.summarize_logging_after_stage()
        synchronize()

        self._valid()
        valid_summary = self.summarize_logging_after_stage()
        synchronize()

        if self.option.train_setting.valid_on_test > 0:
            if epoch % self.option.train_setting.valid_on_test == 0:
                self._test()
                self.summarize_logging_after_stage()
                synchronize()

        if self.lr_scheduler is not None:
            if isinstance(self.lr_scheduler,
                          torch.optim.lr_scheduler.ReduceLROnPlateau):
                key = f'valid/{self._option.lr_scheduler.metric_key}'
                self.lr_scheduler.step(valid_summary[key])
            else:
                self.lr_scheduler.step()
        if self.is_rank0:
            # save models
            self.save_best_model(valid_summary)
            self.backup()
            self.progress_bars[self.STAGE.ALL].update()
            self.rank0_update_logging_after_epoch()

    @property
    def learning_rate(self):
        """ get the learning rate

        :return: double of learning rate
        """
        return self.optimizer.param_groups[0]['lr']

    def set_learning_rate(self, lr):
        """ set the learning rate

        :param lr: double of learning rate
        """
        for group in self.optimizer.param_groups:
            group['lr'] = lr

    # MODEL ADJUSTMENT

    def freeze_and_unfreeze_modules(
        self, names, reset_optimizer: bool = True
    ):
        """ this method freezes the model and then unfreezes modules specified
        by `names`

        :param names: Sequence of module names, should be seekable via `getattr`
        :param reset_optimizer: Bool to reset the optimizer such that only
            unfrozen modules will be updated by the optimizer, default True
        """
        self.unwrapped_model.train(False)
        for parameter in self.unwrapped_model.parameters():
            parameter.requires_grad = False
        for name in names:
            logger.info(f"train {name}")
            name = name.split('.')
            module = self.unwrapped_model
            for n in name:
                module = getattr(module, n)
            module.train(True)
            for parameter in module.parameters():
                parameter.requires_grad = True
        if reset_optimizer:
            if not hasattr(self, 'optimizer'):
                logger.warn('no optimizer to reset')
                return

            lr = self.learning_rate
            self.optimizer = self.option.optimizer.build(self.unwrapped_model)
            self.set_learning_rate(lr)
            logger.info('optimizer reset')

            if not hasattr(self, 'lr_scheduler'):
                return
            self.lr_scheduler = self.option.lr_scheduler.build(self.optimizer)
            logger.info('lr scheduler reset')

    # BATCH LOADING AND MODEL FORWARD BACKWARD

    def load_batch(self):
        """ this method is used as an iterator of the current dataloader, which
        also loads the data from CPU to GPU.
        """
        for batch in self.cur_dataloader:
            # should return a dict of result to use
            for k, v in batch.items():
                # batch[k] = v.cuda(non_blocking=self.is_parallel)
                batch[k] = v.cuda()
            batch_pack = Batch(gt=batch)
            yield batch_pack

    def model_forward_backward(
        self, batch: Batch, backward: bool = False
    ) -> Batch:
        """ This method should define how to perform model forward and
        backward propagation, and update `batch_pack.pred`, `batch_pack.loss`,
        `batch.size`. To make the loss synchronized across gpus,
        call `self.sync_value`.

        :param batch: BatchPack that stores ground truth, prediction, loss
            and batch size
        :param backward: Bool to indicate whether to perform backward
            propagation
        :return: batch, model output, loss and batch_size
        """
        raise NotImplementedError

    @staticmethod
    def sync_value(input_value):
        """ automatically synchronize the result across different gpus. This is
        safe to use in any conditions including single-gpu cases, which will
        return the `input_value` directly.

        :param input_value: value or dict of input value
        :return: reduced value or dict of reduced values
        """
        from pytorch_helper.utils.dist import reduce_value
        return reduce_value(input_value)

    # STAGE ENTRIES

    def train(self, batch_pack: Batch) -> Batch:
        """ this method completes the training with a mini-batch

        :param batch_pack: BatchPack that stores ground truth, prediction, loss
            and batch size
        :return: BatchPack of training result for the mini-batch
        """
        # should return a dict of result to use
        self.optimizer.zero_grad()
        self.model_forward_backward(batch_pack, backward=True)

        if self.option.train_setting.gradient_clip > 0:
            nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                self.option.train_setting.gradient_clip
            )

        if self.option.train_setting.detect_gradient_explosion:
            th = self.option.train_setting.gradient_explosion_threshold
            for parameter in self.model.parameters():
                if parameter.requires_grad:
                    if torch.any(torch.isnan(parameter.grad)):
                        logger.error('nan in grad!')
                        exit(1)
                    elif torch.any(torch.ge(parameter.grad, th)):
                        logger.error('gradient explodes!')
                        exit(1)

        self.optimizer.step()
        return batch_pack

    def _train(self) -> Batch:
        """ this private method has better not be changed. It is designed to
        finish the training over the training dataset and log the result
        properly.

        :return: BatchPack of training result of the last mini-batch
        """
        self.cur_stage = self.STAGE.TRAIN
        self.setup_before_stage()

        with torch.enable_grad():
            for batch_pack in self.load_batch():
                self.train(batch_pack)
                synchronize()
                self.update_logging_in_stage(batch_pack)
                synchronize()
        return batch_pack

    def valid(self, batch: Batch) -> Batch:
        """ this method completes the validation with a mini-batch

        :param batch: BatchPack that stores ground truth, prediction, loss
            and batch size
        :return: BatchPack of validation result for the mini-batch
        """
        # should return a dict of result to use
        with torch.no_grad():
            return self.model_forward_backward(batch)

    def _valid(self) -> Batch:
        """ this private method has better not be changed. It is designed to
        finish the validation over the validation dataset and log the result
        properly.

        :return: dict of validation result of the last mini-batch
        """
        self.cur_stage = self.STAGE.VALID
        self.setup_before_stage()

        with torch.no_grad():
            for batch_pack in self.load_batch():
                self.valid(batch_pack)
                synchronize()
                self.update_logging_in_stage(batch_pack)
                synchronize()
        return batch_pack

    def test(self, batch: Batch) -> Batch:
        """ this method completes the test with a mini-batch

        :param batch: BatchPack that stores ground truth, prediction, loss
            and batch size
        :return: BatchPack of test result for the mini-batch
        """
        # should return a dict of result to use
        with torch.no_grad():
            return self.model_forward_backward(batch)

    def _test(self) -> Batch:
        """ this private method has better not be changed. It is designed to
        finish the test over the test dataset and log the result properly.

        :return: BatchPack of test result of the last mini-batch
        """
        self.cur_stage = self.STAGE.TEST
        self.setup_before_stage()

        with torch.no_grad():
            for batch in self.load_batch():
                self.test(batch)
                synchronize()
                self.update_logging_in_stage(batch)
                synchronize()
        return batch

    ###########################
    # STAGE SETUP AND LOGGING #
    ###########################

    def setup_before_stage(self):
        """ do some setup before a stage, training, validation or testing.
        `self.in_stage_logged` will be set to False. The model will be set for
        training or non-training properly. And `self.cur_dataloader` is also
        switched accordingly. If the stage is `STAGE_TRAIN`,
        `self.current_train_routine` may be updated and makes some changes to
        the model like freezing or unfreezing some modules.
        """
        self.in_stage_logged = False
        if self.cur_stage == self.STAGE.TRAIN:
            self.current_train_routine = \
                self.option.train_setting.get_train_routine(self.epoch)
            self.model.train()
            self.cur_dataloader = self.dataloader.train_loader
            if self.current_train_routine.set_init_lr(self.optimizer):
                if self.is_rank0 and self.epoch > 0:
                    logger.info("Save before applying new routine")
                    self.epoch -= 1
                    self.save_pth(f'epoch_{self.epoch}')
                    self.epoch += 1
            if self.current_train_routine.train_modules is not None:
                self.freeze_and_unfreeze_modules(
                    self.current_train_routine.train_modules,
                    # optimizers like Adam still change the frozen weight
                    # because they are using statistics of gradients
                    reset_optimizer=self.current_train_routine.optimizer_reset
                )
                self.current_train_routine.optimizer_reset = False
        elif self.cur_stage == self.STAGE.VALID:
            self.model.eval()
            self.cur_dataloader = self.dataloader.valid_loader
        elif self.cur_stage == self.STAGE.TEST:
            self.model.eval()
            self.cur_dataloader = self.dataloader.test_loader
        self.setup_logging_before_stage()

    def setup_logging_before_stage(self):
        """ setup logging before start to train/validate/test the model. For
        example, update the progress bar.
        """
        if self.is_rank0:
            self.progress_bars[self.cur_stage].reset(len(self.cur_dataloader))
            self.rank0_setup_logging_before_stage()

    def rank0_setup_logging_before_stage(self):
        """ this method should do some setups only needed on the rank0 process,
        such as tensorboard logging, before a stage begins.
        """
        pass

    def collect_model_output(self, batch):
        model_output = batch.pred
        if isinstance(model_output, dict):
            for key, data in model_output.items():
                if data is not None:
                    self.model_output_dict[key].append(data.cpu().numpy())
        elif isinstance(model_output, torch.Tensor):
            self.model_output_dict['output'].append(
                model_output.cpu().numpy()
            )
        else:
            self.model_output_dict['output'].append(model_output)

    def update_logging_in_stage(self, batch: Batch):
        """ log the result during training/validation/testing, including
        recording the loss with `self.meter`, and call the rank0 process to do
        visualization.

        :param batch: BatchPack instance
        """

        if self.keep_model_output:
            self.collect_model_output(batch)

        if isinstance(batch.loss, dict):
            for k, v in batch.loss.items():
                if v is not None:
                    key = f'{self.cur_stage.value}/{k}-loss'
                    self.meter.record(
                        tag=key, value=v.item(),
                        weight=batch.size,
                        record_op=Meter.RecordOp.APPEND,
                        reduce_op=Meter.ReduceOp.SUM
                    )
                    self.in_stage_meter_keys.add(key)
        else:
            key = f'{self.cur_stage.value}/loss'
            self.meter.record(
                tag=key, value=batch.loss.item(),
                weight=batch.size,
                record_op=Meter.RecordOp.APPEND,
                reduce_op=Meter.ReduceOp.SUM
            )
            self.in_stage_meter_keys.add(key)
        if self.is_rank0:
            self.rank0_update_logging_in_stage(batch)

    def rank0_update_logging_in_stage(self, batch: Batch):
        """ this method should update logging only needed on the rank0 process,
        such as tensorboard logging, during a stage.

        :param batch: BatchPack that stores ground truth, prediction, loss
            and batch size
        """
        self.progress_bars[self.cur_stage].update()
        self.progress_bars[self.cur_stage].refresh()
        if self.tboard is not None:
            if isinstance(batch.loss, dict):
                for k, v in batch.loss.items():
                    self.tboard.add_scalar(
                        f'batch-{self.cur_stage.value}/{k}-loss', v,
                        self.batch_cnt[self.cur_stage]
                    )
                    self.tboard.add_scalar(
                        f'batch/{k}', v, self.batch_cnt[self.STAGE.ALL]
                    )
            else:
                self.tboard.add_scalar(
                    f'batch-{self.cur_stage.value}/loss', batch.loss.item(),
                    self.batch_cnt[self.cur_stage]
                )
                self.tboard.add_scalar(
                    f'batch/loss', batch.loss.item(),
                    self.batch_cnt[self.STAGE.ALL]
                )
            self.batch_cnt[self.cur_stage] += 1
            self.batch_cnt[self.STAGE.ALL] += 1

    def summarize_logging_after_stage(self) -> OrderedDict:
        """ get the summary of the result over the whole training/validation/
        test dataset.

        :return: dict of stage summary
        """
        summary = OrderedDict()

        for key in sorted(list(self.in_stage_meter_keys)):
            if key.startswith(self.cur_stage.value):
                summary[key] = self.meter.mean(key)

        for k, v in summary.items():
            tag = f'epoch-{k}'
            self.meter.record(
                tag=tag, value=v,
                record_op=Meter.RecordOp.APPEND,
                reduce_op=Meter.ReduceOp.STORE
            )
            if self.is_rank0:
                logger.info(f'{tag} = {v}')

        if self.is_rank0:
            if self.tboard is not None:
                for k, v in summary.items():
                    self.tboard.add_scalar(f'epoch-{k}', v, self.epoch)
            self.rank0_update_logging_after_stage(summary)

        return summary

    def rank0_update_logging_after_stage(self, summary: dict):
        """ this method should update logging only needed on the rank0
        process, such as tensorboard logging, after a stage

        :param summary: dict of stage summary
        :return:
        """
        pass

    def rank0_update_logging_after_epoch(self):
        """ this method should update logging only needed on the rank0 process.
        """
        pass
