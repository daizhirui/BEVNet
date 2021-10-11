from dataclasses import dataclass
from typing import Optional
from typing import Tuple

from pytorch_helper.settings.options import ModelOption
from pytorch_helper.settings.options import OptionBase
from pytorch_helper.settings.options import TaskMode
from pytorch_helper.settings.options import TaskOption
from pytorch_helper.settings.options import TrainRoutine
from pytorch_helper.settings.options import TrainSettingOption
from pytorch_helper.settings.options.descriptors import AutoConvertDescriptor
from pytorch_helper.settings.spaces import Spaces
from pytorch_helper.utils.io import load_pth
from pytorch_helper.utils.log import get_logger
from .test_metric_option import TestMetricOption

__all__ = ['BEVNetTaskOption', 'BEVNetOption', 'BEVNetTrainRoutine']

logger = get_logger(__name__)


@dataclass()
class SharedEncoderOption(OptionBase):
    enable: bool
    encoder_type: str = None
    branches: Tuple = None

    def build(self):
        if self.enable:
            assert self.encoder_type, \
                f'encoder type of the shared encoder is {self.encoder_type}'
            assert self.branches, \
                f'branches that will share the encoder is {self.branches}'
            from models.encoders import build_encoder
            return build_encoder(self.encoder_type, pretrained=True)
        else:
            self.branches = ()
            return None


@dataclass()
class BEVDecoderOption(OptionBase):
    decode_feet_branch: bool
    decode_head_branch: bool
    magnitude_scale: int
    head_heights: Tuple[int] = (1.8, 1.7, 1.5, 1.2)
    head_bev_attention: bool = True


@dataclass()
class BEVNetOption(ModelOption):

    def __post_init__(self):
        assert self.ref.lower().startswith('bevnet'), \
            f"BEVNetOption is not for {self.ref}"

        self.head_branch_option = ModelOption.from_dict(
            self.kwargs['head_branch_option']
        )
        self.feet_branch_option = ModelOption.from_dict(
            self.kwargs['feet_branch_option']
        )
        self.pose_branch_option = ModelOption.from_dict(
            self.kwargs['pose_branch_option']
        )
        self.bev_decoder_option = BEVDecoderOption.from_dict(
            self.kwargs['bev_decoder_option']
        )
        self.share_encoder_option = SharedEncoderOption.from_dict(
            self.kwargs['share_encoder_option']
        )

        self.head_branch_enable = self.head_branch_option is not None
        self.feet_branch_enable = self.feet_branch_option is not None

        # check if the configuration is correct
        assert self.feet_branch_enable or self.head_branch_enable, \
            "feet branch and head branch cannot be disabled together"
        assert self.pose_branch_option is not None, \
            "pose branch option cannot be created, option file required"
        assert self.bev_decoder_option is not None, \
            "bev decoder branch option cannot be created, option file required"
        if self.bev_decoder_option.decode_head_branch:
            assert self.head_branch_enable, \
                "head branch is required by the bev decoder"
        if self.bev_decoder_option.decode_feet_branch:
            assert self.feet_branch_enable, \
                "feet branch is required by the bev decoder"

    @staticmethod
    def optional_build(option: Optional[ModelOption]):
        if option:
            return option.build()
        else:
            return None, None

    def build(self):
        net_head, _ = self.optional_build(self.head_branch_option)
        net_feet, _ = self.optional_build(self.feet_branch_option)
        net_pose, _ = self.pose_branch_option.build()
        shared_encoder = self.share_encoder_option.build()

        net_kwargs = dict(
            head_branch=net_head,
            feet_branch=net_feet,
            pose_branch=net_pose,
            shared_encoder=shared_encoder,
            branches_share_encoder=self.share_encoder_option.branches,
            magnitude_scale=self.bev_decoder_option.magnitude_scale,
            head_heights=self.bev_decoder_option.head_heights
        )
        if self.ref == 'BEVNetFeetOnly':
            del net_kwargs['head_branch']
            del net_kwargs['head_heights']
        if self.ref == 'BEVNetNoAttention':
            del net_kwargs['head_heights']

        model = Spaces.build(Spaces.NAME.MODEL, self.ref, net_kwargs)
        logger.info(f'Build {type(model).__name__}')

        state_dict = None
        if self.pth_available:
            state_dict = load_pth(self.pth_path)
            model.load_state_dict(state_dict['model'])
            epoch = state_dict.get('epoch', 'NA')
            logger.info(f'Load model state from epoch {epoch}')
        return model, state_dict


@dataclass()
class BEVNetTrainRoutine(TrainRoutine):
    use_inferred_pose: bool = True
    loss_weight_changes: dict = None


class BEVNetTrainSettingOption(TrainSettingOption):

    def __post_init__(self):
        self.train_routines = [
            BEVNetTrainRoutine(**r) for r in self.train_routines
        ]
        self.train_routines.sort(key=lambda x: x.epochs)

    def get_train_routine(self, epoch):
        for r in self.train_routines:
            if epoch < r.epochs:
                return r
        raise ValueError(f'train routine for epoch {epoch} is missing')


@Spaces.register(Spaces.NAME.TASK_OPTION, ['BEVNetTask', 'IV2BEVTask'])
class BEVNetTaskOption(TaskOption):
    model = AutoConvertDescriptor(BEVNetOption.from_dict)
    train_setting = AutoConvertDescriptor(BEVNetTrainSettingOption.from_dict)

    def __post_init__(self, mode, is_distributed):
        mode = TaskMode(mode)
        if mode != TaskMode.TRAIN:
            assert self.test_option, 'Please specify a test option file'
            self.test_option = self.load_option(
                self.test_option, TestMetricOption
            )
        super(BEVNetTaskOption, self).__post_init__(mode, is_distributed)
