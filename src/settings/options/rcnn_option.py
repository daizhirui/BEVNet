from dataclasses import dataclass

import matplotlib
import os
import pickle
import random
import torch
from cv2 import cv2
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from matplotlib import pyplot as plt
from typing import Union

from datasets.cityuhk.dataset import CityUHKBEV
from pytorch_helper.settings.options import ModelOption
from pytorch_helper.settings.options import TaskOption
from pytorch_helper.utils.io import imsave
from pytorch_helper.utils.log import get_logger
from pytorch_helper.utils.log import pbar
from pytorch_helper.settings.spaces import Spaces

from .test_metric_option import TestMetricOption

__all__ = ['RCNNTaskOption']
matplotlib.use('Agg')

logger = get_logger(__name__)


@Spaces.register(Spaces.NAME.TASK_OPTION, ['RCNNTask', 'RCNN2BEVTask'])
@dataclass()
class RCNNTaskOption(TaskOption):
    rcnn_config_file: str = None
    pose_net: Union[dict, ModelOption] = None

    def __post_init__(self, mode, is_distributed):
        super(RCNNTaskOption, self).__post_init__(mode, is_distributed)
        assert self.rcnn_config_file and os.path.isfile(self.rcnn_config_file), \
            f'{self.rcnn_config_file} is not a file!'
        self.dataset_path_det = os.path.join(
            self.dataset_path, 'det'
        )
        os.makedirs(self.dataset_path_det, exist_ok=True)
        self.aspect_ratio = 3
        self.confidence_threshold = 0.5

        self.pose_net = self.load_option(self.pose_net, ModelOption)
        self.test_option = self.load_option(self.test_option, TestMetricOption)

    def setup_cfg(self):
        # load config from file and command-line arguments
        cfg = get_cfg()
        cfg.merge_from_file(self.rcnn_config_file)

        if self.train:
            cfg.DATASETS.TRAIN = ("cityuhk-det-train",)
            cfg.DATASETS.TEST = ("cityuhk-det-test",)
            DatasetCatalog.register(
                "cityuhk-det-train",
                lambda: self.prepare_cityuhk_det(train=True)
            )
            DatasetCatalog.register(
                "cityuhk-det-test",
                lambda: self.prepare_cityuhk_det(train=False)
            )
            MetadataCatalog.get("cityuhk-det-train").set(
                thing_classes=["PERSON"])
            MetadataCatalog.get("cityuhk-det-test").set(
                thing_classes=["PERSON"])
            self.vis_dataset()

        # Number of data loading threads
        cfg.DATALOADER.NUM_WORKERS = 4
        # Number of images per batch across all machines.
        cfg.SOLVER.IMS_PER_BATCH = 8
        cfg.SOLVER.BASE_LR = 0.0125  # pick a good LearningRate
        cfg.SOLVER.MAX_ITER = 5000  # No. of iterations
        # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
        # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = self.confidence_threshold
        # No. of iterations after which the Validation Set is evaluated.
        cfg.TEST.EVAL_PERIOD = 500
        cfg.OUTPUT_DIR = os.path.join(self.output_path_pth)
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        cfg.freeze()
        return cfg

    @staticmethod
    def get_bbox(n, feet_coords, head_coords, angle, aspect_ratio):
        feet_coords = feet_coords[:2, :int(n)]
        head_coords = head_coords[:2, :int(n)]
        gt_w = (feet_coords[1] - head_coords[1]) / (
            aspect_ratio * torch.cos(angle))
        gt_box = torch.zeros(int(n), 4)
        gt_box[:, 0] = feet_coords[0] - 0.5 * gt_w  # left
        gt_box[:, 1] = head_coords[1]  # top
        gt_box[:, 2] = feet_coords[0] + 0.5 * gt_w  # right
        gt_box[:, 3] = feet_coords[1]  # bottom
        return gt_box

    def prepare_cityuhk_det(self, train):
        phase = 'train' if train else 'test'
        anno_path = os.path.join(self.dataset_path_det, f'{phase}_list.pkl')
        if os.path.exists(anno_path):
            return pickle.load(open(anno_path, 'rb'))

        dataloaders = self.dataloader.build()
        if train:
            dset = CityUHKBEV(
                root=dataloaders.root,
                datalist=dataloaders.train_datalist,
                keys=dataloaders.keys,
                use_augment=False
            )
        else:
            dset = CityUHKBEV(
                root=dataloaders.root,
                datalist=dataloaders.valid_datalist,
                keys=dataloaders.keys,
                use_augment=False
            )
        dset.do_normalization = False

        data_list = []
        logger.info('Prepare dataset for detection methods')
        for i, sample in enumerate(pbar(dset)):
            im = sample['image'].permute(1, 2, 0).numpy()
            im_path = os.path.join(self.dataset_path_det, f'{phase}_{i}.png')
            if not os.path.isfile(im_path):
                plt.imsave(im_path, im)
            h, w, _ = im.shape

            n_annos = sample['num_annotations']
            feet_annos = sample['feet_annotation']
            head_annos = sample['head_annotation']
            camera_angle = sample['camera_angle']
            bboxes = self.get_bbox(
                n_annos, feet_annos, head_annos, camera_angle, self.aspect_ratio
            )

            data_list.append({
                'file_name': im_path,
                'height': h,
                'width': w,
                'image_id': i,
                'annotations': [{
                    'bbox': bbox.tolist(),
                    'bbox_mode': 0,
                    'category_id': 0
                } for bbox in bboxes]
            })

        if not os.path.exists(anno_path):
            with open(anno_path, 'wb') as file:
                pickle.dump(data_list, file)
        return data_list

    def vis_dataset(self, n=3):
        dataset_dicts = self.prepare_cityuhk_det(train=True)
        for k, d in enumerate(random.sample(dataset_dicts, n)):
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(
                img[:, :, ::-1], MetadataCatalog.get("cityuhk-det-train")
            )
            vis = visualizer.draw_dataset_dict(d)
            imsave(
                os.path.join(self.dataset_path_det, 'temp', f"ex{k}.png"),
                vis.get_image()[:, :, ::-1]
            )
