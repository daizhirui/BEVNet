from collections import OrderedDict
from collections import defaultdict

import h5py
import numpy as np
import os
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer
from detectron2.engine import launch
from detectron2.modeling import build_model
from detectron2.structures import Boxes
from detectron2.structures import Instances
from detectron2.utils.visualizer import Visualizer

from models import BEVTransform
from models.metrics.kernels import GaussianKernel
from pytorch_helper.launcher.launcher import LauncherTask
from pytorch_helper.task import Batch
from pytorch_helper.utils.dist import get_rank
from pytorch_helper.utils.io import make_dirs
from pytorch_helper.utils.io import save_dict_as_csv
from pytorch_helper.utils.log import get_datetime
from pytorch_helper.utils.log import get_logger
from pytorch_helper.utils.log import pbar
from pytorch_helper.utils.meter import Meter
from pytorch_helper.settings.spaces import Spaces

logger = get_logger(__name__)


@Spaces.register(Spaces.NAME.TASK, ['RCNNTask', 'RCNN2BEVTask'])
class RCNNTask(LauncherTask):

    def __init__(self, task_option):
        super(RCNNTask, self).__init__()
        self._option = task_option
        self.cuda_ids = task_option.cuda_ids
        self.rank = get_rank()

        self.trainer = None
        if not self._option.train:
            self.output_path_test_net_output = os.path.join(
                self._option.output_path_test, 'net_output'
            )
            make_dirs(self._option.output_path_test)
            make_dirs(self.output_path_test_net_output)

            torch.cuda.set_device(self.cuda_ids[0])

            cfg = self.option.setup_cfg()
            self.model = build_model(cfg)
            checkpointer = DetectionCheckpointer(self.model)
            checkpointer.load(cfg.MODEL.WEIGHTS)
            self.loss_fn = self.option.loss.build()

            self.use_inferred_pose = not self.option.test_option.pose_oracle
            if self.use_inferred_pose:
                self.pose_net = self.option.pose_net.build()[0]
                self.pose_net.cuda()
                self.pose_net.eval()
            self.bev_transform = BEVTransform()

            sigma = 5
            ks = sigma * 6 + 1
            self.gaussian_conv = GaussianKernel(ks, sigma, channels=1)
            self.gaussian_conv.cuda()

            self.dataloader = self.option.dataloader.build()
            self.test_loader = self.dataloader.test_loader
            self.test_loader.dataset.do_normalization = False

            self.meter = Meter()
            self.in_stage_meter_keys = set()
            self.model_output_dict = defaultdict(list)

    @property
    def is_rank0(self):
        return self.rank == 0

    @property
    def is_distributed(self):
        return False

    @property
    def option(self):
        return self._option

    def train(self):
        self.trainer = DefaultTrainer(self.option.setup_cfg())
        self.trainer.resume_or_load(resume=self.option.resume)
        self.trainer.train()

    def run(self):
        if self._option.train:
            if len(self.cuda_ids) > 1:
                launch(
                    main_func=self.train,
                    num_gpus_per_machine=len(self.cuda_ids),
                    dist_url=f'tcp://localhost:{int(os.environ["DDP_PORT"])}'
                )
            else:
                self.train()
        else:
            self.model.eval()
            self.pose_net.eval()
            for batch in pbar(self.test_loader, desc='Test'):
                with torch.no_grad():
                    result = self.model_forward(batch)
                self.update_logging_in_stage(result)

            summary = self.summarize_logging_after_stage()

            path = os.path.join(self._option.output_path_test,
                                'test-summary.csv')
            save_dict_as_csv(path, summary)

            if self.option.test_option.save_model_output:
                path = os.path.join(self._option.output_path_test,
                                    f'model-output.h5')
                logger.info(f'Saving model output to {path}')
                h5file = h5py.File(path, 'w')
                for key, value in summary.items():
                    h5file.attrs[key] = value
                h5file.attrs['summary-ordered-keys'] = list(summary.keys())
                h5file.attrs['datetime_test'] = get_datetime()
                for name, data in self.model_output_dict.items():
                    logger.info(f'Saving {name}')
                    data = np.concatenate(data, axis=0)
                    h5file.create_dataset(
                        name, data.shape, data.dtype, data, compression='gzip'
                    )
                h5file.close()

    def model_forward(self, batch: dict):
        for k, v in batch.items():
            batch[k] = v.cuda()

        images = batch['image']
        gt_bev_map = batch['bev_map']
        bs, _, height, width = gt_bev_map.shape
        inputs = [
            {'image': im, 'height': height, 'width': width}
            for im in torch.flip(images, [1]) * 255
        ]
        outputs = self.model(inputs)
        predictions = [output['instances'] for output in outputs]
        person_bboxes = [
            pred.pred_boxes.tensor[pred.pred_classes == 0]
            for pred in predictions
        ]

        pred = dict(outputs=outputs, person_bboxes=person_bboxes)
        if self.use_inferred_pose:
            pose_inputs = self.dataloader.normalize(images)
            pred_pose = self.pose_net(pose_inputs.cuda())
            pred.update(**pred_pose)

        result = Batch(gt=batch, pred=pred)
        n_coords = [len(bboxes) for bboxes in person_bboxes]
        bs, _, height, width = result.gt['bev_map'].size()

        # process outputs from detection model
        pred_feet_map = torch.zeros((bs, 1, height, width)).cuda()
        pred_head_map = torch.zeros((bs, 1, height, width)).cuda()
        feet_pixels = torch.zeros(bs, 3, max(n_coords)).cuda()
        feet_pixels[:, 2] = 1
        for i, bboxes in enumerate(person_bboxes):
            # feet position = center of bottom bbox border
            feet_pixels[i, 0, :len(bboxes)] = (bboxes[:, 0] + bboxes[:, 2]) / 2
            feet_pixels[i, 1, :len(bboxes)] = bboxes[:, 3]

            u = feet_pixels[i, 0, :len(bboxes)].long()
            v = feet_pixels[i, 1, :len(bboxes)].long()
            mask = (0 <= v) & (0 <= u) & (v < height) & (u < width)
            u = u[mask]
            v = v[mask]
            pred_feet_map[i, 0, v, u] = 1.

            u = ((bboxes[:, 0] + bboxes[:, 2]) / 2).long()
            v = bboxes[:, 1].long()
            mask = (0 <= v) & (0 <= u) & (v < height) & (u < width)
            u = u[mask]
            v = v[mask]
            pred_head_map[i, 0, v, u] = 1.
        pred_feet_map = self.gaussian_conv(pred_feet_map)
        pred_head_map = self.gaussian_conv(pred_head_map)
        result.pred['feet_map'] = pred_feet_map
        result.pred['head_map'] = pred_head_map

        # image view detection to world coordinates
        im_size = (height, width)
        key = 'pred' if self.use_inferred_pose else 'gt'
        cam_h = getattr(result, key)['camera_height']
        cam_a = getattr(result, key)['camera_angle']
        camera_fu = result.gt['camera_fu']
        camera_fv = result.gt['camera_fv']
        i2w_mats, scales, centers = self.bev_transform.get_bev_param(
            im_size, cam_h, cam_a, camera_fu, camera_fv, w2i=False
        )
        result.pred['bev_scale'] = scales
        result.pred['bev_center'] = centers

        pred_world_coords_homo = self.bev_transform.image_coord_to_world_coord(
            feet_pixels, i2w_mats
        )
        pred_world_coords = [
            coord[:2, :n].cpu()
            for coord, n in zip(pred_world_coords_homo, n_coords)
        ]
        # evaluate the region of interest inside the BEV map only
        pred_bev_coords = self.bev_transform.world_coord_to_bev_coord(
            im_size, pred_world_coords_homo, scales, centers
        )
        pred_bev_map = torch.zeros((bs, 1, height, width)).cuda()

        # filter predicted coords
        pred_coords = []
        for i, (world_coords, bev_coords, n_annos) in enumerate(
            zip(pred_world_coords, pred_bev_coords, n_coords)
        ):
            world_coords = world_coords[:2, :int(n_annos)]
            bev_coords = bev_coords[:2, :int(n_annos)]
            roi = (bev_coords[0] >= 0) & (bev_coords[0] < width) & \
                  (bev_coords[1] >= 0) & (bev_coords[1] < height)
            pred_coords.append(world_coords[:, roi])
            for u, v in bev_coords[:, roi].int().T:
                pred_bev_map[i, 0, v.item(), u.item()] = 1
        pred_bev_map = self.gaussian_conv(pred_bev_map)
        result.pred['bev_map'] = pred_bev_map

        # loss
        result.loss = self.loss_fn(result.pred, result.gt)
        result.size = bs

        return result

    def save_visualization(self, result):
        def get_path(x):
            return os.path.join(self.output_path_test_net_output, x)

        images = result.gt['image']
        bs, _, h, w = images.size()
        for i in range(bs):
            im = images[i].cpu().permute(1, 2, 0).numpy() * 255
            scene_id = result.gt['scene_id'][i].int().item()
            image_id = result.gt['image_id'][i].int().item()
            prefix = f'scene{scene_id:02d}-{image_id}'
            visualizer = Visualizer(im)

            # prediction
            # detection
            instances = result.pred['outputs'][i]['instances'].to('cpu')
            # keep people class
            instances = instances[instances.pred_classes == 0]
            vis_output = visualizer.draw_instance_predictions(instances)
            vis_output.save(get_path(prefix + '-detection-pred.png'))

            # gt
            # pseudo-detection
            num_annotation = result.gt['num_annotations'][i].long()
            feet = result.gt['feet_annotation'][i, :, :num_annotation]
            head = result.gt['head_annotation'][i, :, :num_annotation]
            camera_angle = result.gt['camera_angle'][i]
            box_w = (feet[1] - head[1]) / (3 * camera_angle.cos())
            box = torch.zeros(int(num_annotation), 4)

            box[:, 0] = feet[0] - 0.5 * box_w  # left
            box[:, 1] = head[1]  # top
            box[:, 2] = feet[0] + 0.5 * box_w  # right
            box[:, 3] = feet[1]  # bottom

            gt_instances = Instances(
                (h, w), pred_boxes=Boxes(box),
                scores=torch.ones(len(box)),
                pred_classs=torch.zeros(len(box)).long()
            )
            visualizer = Visualizer(im)
            vis_output = visualizer.draw_instance_predictions(gt_instances)
            vis_output.save(get_path(prefix + '-detection-gt.png'))

    def update_logging_in_stage(self, result):
        loss = result.loss
        for k, v in loss.items():
            if v is not None:
                key = f'test/{k}-loss'
                self.meter.record(
                    tag=key, value=v.item(), weight=result.size,
                    record_op=Meter.RecordOp.APPEND,
                    reduce_op=Meter.ReduceOp.SUM
                )
                self.in_stage_meter_keys.add(key)

        if self.option.test_option.save_im:
            self.save_visualization(result)
        if self.option.test_option.save_model_output:
            for key in ['scene_id', 'image_id']:
                self.model_output_dict.setdefault(key, list()).append(
                    result.gt[key].cpu().numpy()
                )
            for key, data in result.pred.items():
                if isinstance(data, torch.Tensor):
                    self.model_output_dict.setdefault(key, list()).append(
                        data.cpu().numpy()
                    )

    def summarize_logging_after_stage(self):
        summary = OrderedDict()
        summary['name'] = self.option.name
        summary['datetime'] = self.option.datetime
        summary['epoch'] = 'NA'
        summary['pth_file'] = 'NA'
        for key in sorted(list(self.in_stage_meter_keys)):
            summary[key] = self.meter.mean(key)
        return summary

    def backup(self, immediate, resumable):
        if self.trainer:
            self.trainer.checkpointer.save(f'iter_{self.trainer.iter}')
