import argparse
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np
import torch.cuda
from matplotlib import pyplot as plt

from models import BEVTransform
from models import DensityCluster
from models import IndividualDistance
from pytorch_helper.settings.spaces import Spaces
from pytorch_helper.utils.image import to_heatmap
from pytorch_helper.utils.io import imsave
from pytorch_helper.utils.io import load_yaml
from pytorch_helper.utils.io import save_dict_as_csv
from pytorch_helper.utils.log import get_logger
from pytorch_helper.utils.log import pbar
from pytorch_helper.utils.meter import Meter
from pytorch_helper.utils.pre_pytorch_init import set_cuda_visible_devices
from settings.options.test_metric_option import TestMetricOption
from utils.visualize import save_density_map

logger = get_logger(__name__)


@dataclass()
class Args:
    task_option_file: str
    test_option_file: str
    model_output_file: str
    output_path: str
    output_csv: str
    dataset_path: str
    use_gpu: int


@dataclass()
class Result:
    gt: Any
    pred: Any
    size: int
    loss: Any = None

    def __post_init__(self):
        self.metric_results = dict()


class MetricRunner:
    def __init__(self, args: Args):
        task_option = load_yaml(args.task_option_file)
        task_option['is_distributed'] = False
        task_option['mode'] = 'test'
        task_option['resume'] = False
        if args.dataset_path:
            task_option['dataset_path'] = args.dataset_path
        task_option = Spaces.build(
            Spaces.NAME.TASK_OPTION, task_option['ref'], task_option
        )

        if args.test_option_file:
            self.test_option = TestMetricOption.from_file(
                args.test_option_file
            )
            task_option.test_option = self.test_option
        else:
            self.test_option = task_option.test_option
        assert self.test_option, 'No test option is specified.'

        logger.info(f'Load model output from {args.model_output_file}')
        self.net_output = h5py.File(args.model_output_file, 'r')

        keys = task_option.dataloader.kwargs['keys']
        if isinstance(keys, str):
            keys = keys.split(',')
        keys = set(keys)
        keys.add('bev_scale')
        keys.add('bev_center')
        task_option.dataloader.kwargs['keys'] = ','.join(keys)
        self.dataloader = task_option.dataloader.build()
        self.bev_transform = BEVTransform()

        # load metrics
        self.metrics = [
            m.build() for m in self.test_option.metrics
        ]
        self.metrics = {type(m).__name__: m for m in self.metrics}

        # miscellaneous
        self.meter = Meter()
        self.metric_meter_keys = set()
        self.scene_metric_keys = set()
        if args.output_path:
            self.output_path_test = args.output_path
        else:
            self.output_path_test = os.path.dirname(args.model_output_file)
        torch.cuda.set_device(args.use_gpu)

    def run_metrics(self):
        cnt = 0
        test_loader = self.dataloader.test_loader
        for batch in pbar(test_loader, desc='Metric'):
            for k, v in batch.items():
                batch[k] = v.cuda()
            bs = batch['bev_map'].size(0)

            batch['image'] = self.dataloader.denormalize(batch['image'])
            batch['iv_roi'] = self.bev_transform.get_iv_roi(
                batch['bev_map'].size(), batch['camera_height'],
                batch['camera_angle'], batch['camera_fu'], batch['camera_fv']
            )

            pred = {
                k: torch.tensor(self.net_output[k][cnt:cnt + bs]).cuda()
                for k in self.net_output
            }

            pred['iv_roi'] = self.bev_transform.get_iv_roi(
                pred['bev_map'].size(), pred['camera_height'],
                pred['camera_angle'], batch['camera_fu'], batch['camera_fv']
            )

            result = Result(gt=batch, pred=pred, size=bs)

            dist_metric_result = self.run_individual_distance(result)
            self.run_density_cluster(result, dist_metric_result)

            self.save_visualization(result)

            cnt += bs

        self.save_global_and_individual_risk()
        self.save_scene_metrics()

        summary = OrderedDict()
        for key in self.net_output.attrs['summary-ordered-keys']:
            summary[key] = self.net_output.attrs[key]
        metric_summary = self.meter.means(self.metric_meter_keys)
        for key in sorted(list(metric_summary.keys())):
            summary[key] = metric_summary[key]
            logger.info(f'{key} = {metric_summary[key]}')

        path = os.path.join(self.output_path_test, f'test-summary.csv')
        save_dict_as_csv(path, summary)

        return summary

    def run_individual_distance(self, result: Result) -> dict:
        name = IndividualDistance.__name__
        dist_metric_result = None
        if name in self.metrics:
            dist_metric_result = self.metrics[name](result.pred, result.gt)
            # record metric result
            for k, v in dist_metric_result['summary'].items():
                self.meter.record(
                    tag=k, value=v,
                    record_op=Meter.RecordOp.EXTEND,
                    reduce_op=Meter.ReduceOp.STORE
                )
                self.metric_meter_keys.add(k)
        result.metric_results[name] = dist_metric_result
        return dist_metric_result

    def run_density_cluster(self, result: Result, dist_metric_result: dict):
        name = DensityCluster.__name__
        cluster_metric_result = None

        if name not in self.metrics:
            return cluster_metric_result

        # run metric
        cluster_metric_result = self.metrics[name](
            result.pred, result.gt, dist_metric_result
        )
        result.metric_results[name] = cluster_metric_result
        batch = result.gt

        # record by scene
        for i, image_id in enumerate(batch['image_id'].int().tolist()):
            scene_id = batch['scene_id'][i].int().item()
            prefix = f'scene{scene_id:02d}'
            self.meter.record(
                tag='scene_id', value=scene_id,
                record_op=Meter.RecordOp.APPEND,
                reduce_op=Meter.ReduceOp.STORE
            )
            self.meter.record(
                tag='image_id', value=image_id,
                record_op=Meter.RecordOp.APPEND,
                reduce_op=Meter.ReduceOp.STORE
            )

            self.meter.record(
                tag=f'{prefix}-image',
                value=batch['image'][[i]].cpu().numpy(),
                record_op=Meter.RecordOp.APPEND,
                reduce_op=Meter.ReduceOp.SUM
            )

            # scene risk and individual risk
            for j in ['pred', 'gt']:
                for k in ['risk_map_bev', 'risk_map_iv']:
                    v = cluster_metric_result[j][k]
                    tag = f'{prefix}-{k}-{j}'
                    self.meter.record(
                        tag=tag, value=v[i, 0].cpu().numpy(),
                        record_op=Meter.RecordOp.APPEND,
                        reduce_op=Meter.ReduceOp.SUM
                    )
                    self.scene_metric_keys.add(tag)

                global_risk = cluster_metric_result[j]['global_risk'][i]
                self.meter.record(
                    tag=f'global_risk-{j}',
                    value=global_risk,
                    record_op=Meter.RecordOp.APPEND,
                    reduce_op=Meter.ReduceOp.STORE
                )

                indiv_risks = cluster_metric_result[j]['individual_risks'][i]
                if len(indiv_risks) == 0:
                    indiv_risks = [(0, 0, 0)]
                else:
                    self.meter.record(
                        tag=f'individual_risks-{j}',
                        value=cluster_metric_result[j]['individual_risks'][i],
                        record_op=Meter.RecordOp.APPEND,
                        reduce_op=Meter.ReduceOp.STORE
                    )
                indiv_risks = np.array(indiv_risks)[:, 2]
                area = cluster_metric_result[j]['area'][i]
                path = f'{prefix}-{image_id}-individual_risk_hist-{j}'
                path = os.path.join(self.output_path_test, name, path)
                self.save_individual_risk_hist(
                    path, global_risk, indiv_risks, area
                )

        # record metric summary: loss and iou
        for k, v in cluster_metric_result['summary']['loss'].items():
            tag = f'{k}-loss'
            if v.numel() == 1:
                self.meter.record(
                    tag=tag, value=v.item(), weight=result.size,
                    record_op=Meter.RecordOp.APPEND,
                    reduce_op=Meter.ReduceOp.SUM
                )
            else:
                self.meter.record(
                    tag=tag, value=v.cpu().numpy(),
                    record_op=Meter.RecordOp.EXTEND,
                    reduce_op=Meter.ReduceOp.SUM
                )
            self.metric_meter_keys.add(tag)
        # record iou
        self.meter.record(
            tag='iou', value=cluster_metric_result['summary']['iou'],
            record_op=Meter.RecordOp.EXTEND,
            reduce_op=Meter.ReduceOp.SUM
        )
        self.metric_meter_keys.add('iou')
        return cluster_metric_result

    def save_individual_risk_hist(
        self, path, global_risk, indiv_risks, area
    ):
        if not self.test_option.save_im:
            return

        plt.hist(indiv_risks, bins=40, range=(1, 40))
        plt.xlabel('Individual Risk')
        plt.ylabel('Count')
        plt.title(
            f'Global risk: {global_risk:.2e}, '
            rf'area: {area:.2e} $m^2$'
        )
        plt.tight_layout()
        imsave(path, plt.gcf())
        plt.close()

    def save_global_and_individual_risk(self):
        for j in ['pred', 'gt']:
            tag = f'global_risk-{j}'
            if tag in self.meter:
                plt.hist(np.array(self.meter[tag].data))
                plt.xlabel('Risk')
                plt.ylabel('Count')
                plt.savefig(os.path.join(self.output_path_test, tag))
                plt.close()
                # save scene risks
                filename = os.path.join(
                    self.output_path_test, tag + '.csv'
                )
                np.savetxt(
                    fname=filename,
                    X=np.vstack([
                        np.array(self.meter['scene_id'].data),
                        np.array(self.meter['image_id'].data),
                        np.array(self.meter[tag].data),
                        np.array(self.meter[f'global_risk-{j}'].data)
                    ]).T,
                    delimiter=',',
                    header="scene_id,image_id,global-risk,violation-cnt",
                )

            tag = f'individual_risks-{j}'
            if tag in self.meter:
                indiv_risks = np.concatenate(self.meter[tag].data, axis=0)
                plt.hist(indiv_risks[:, -1])
                plt.xlabel('Risk')
                plt.ylabel('Count')
                plt.savefig(os.path.join(self.output_path_test, tag))
                plt.close()
                # save individual risks
                arr = []
                for s, i, risks in zip(self.meter['scene_id'].data,
                                       self.meter['image_id'].data,
                                       self.meter[tag].data
                                       ):
                    if len(risks) == 0:
                        continue
                    scene_ids = np.ones((len(risks), 1)) * s
                    image_ids = np.ones((len(risks), 1)) * i
                    arr.append(np.concatenate(
                        [scene_ids, image_ids, risks], axis=1
                    ))
                filename = os.path.join(self.output_path_test, tag + '.csv')
                arr = np.concatenate(arr, axis=0)
                np.savetxt(
                    filename, X=arr, delimiter=',',
                    header='scene_id,image_id,u,v,risk'
                )

    def save_scene_metrics(self):
        if not self.test_option.save_im:
            return
        output_dir = os.path.join(
            self.output_path_test, DensityCluster.__name__, 'scene'
        )
        for key, metric in self.meter.means(self.scene_metric_keys).items():
            path1 = os.path.join(output_dir, key)
            if 'risk_map_iv' in key:
                prefix = key.split('-')[0]
                path2 = os.path.join(output_dir, f'{key}-syn')
                image = self.meter.mean(f'{prefix}-image')[0].transpose(1, 2, 0)

                save_density_map(
                    path1, metric, path2, image
                )
            else:
                imsave(path1, to_heatmap(metric))

    def save_visualization(self, result: Result):
        if not self.test_option.save_im:
            return

        def get_path(x):
            return os.path.join(self.output_path_test, x)

        for i in range(result.gt['image'].size(0)):
            scene_id = result.gt['scene_id'][i].int()
            image_id = result.gt['image_id'][i].int()
            prefix = f'scene{scene_id:02d}-{image_id}'

            # input
            input_im = result.gt['image'][i].cpu().numpy().transpose(1, 2, 0)

            # project detection of head and feet onto images, synthesize and
            # save the result
            for j in ['pred', 'gt']:
                # save bev graph
                name = IndividualDistance.__name__
                metric_result = result.metric_results.get(name, None)
                if isinstance(metric_result, dict) and 'graph' in metric_result:
                    filename = os.path.join(name, f'{prefix}-graph-{j}')
                    metric_result['graph'][j].savefig(get_path(filename))

                # save density cluster results
                name = DensityCluster.__name__
                if name in result.metric_results:
                    metric_result = result.metric_results[name]
                    # bev
                    key = f'risk_map_bev'
                    filename = os.path.join(name, f'{prefix}-{key}-{j}')
                    path = get_path(filename)
                    image = metric_result[j][key][i, 0].cpu().numpy()
                    save_density_map(path, image)

                    # iv
                    key = f'risk_map_iv'
                    filename = f'{prefix}-{key}-{j}'
                    filename1 = os.path.join(name, filename)
                    path1 = get_path(filename1)
                    filename2 = os.path.join(name, f'{filename}-syn')
                    path2 = get_path(filename2)
                    density_map = metric_result[j][key][i, 0].cpu().numpy()
                    iv_roi = getattr(result, j)['iv_roi'][i, 0].cpu().numpy()
                    texts = None
                    if self.test_option.label_risk:
                        texts = [
                            (u, v, f'{r:.2f}')
                            for u, v, r in metric_result['individual_risks'][i]
                        ]
                        global_risk = metric_result[j]['global_risk'][i]
                        texts.append(
                            (5, 15, f'Global risk: {global_risk:.3f}')
                        )
                    save_density_map(
                        path1, density_map, path2, input_im, iv_roi, texts
                    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task-option-file', required=True, type=str,
        help='Path to the file of bevnet task option, used to create dataloader'
    )
    parser.add_argument(
        '--test-option-file', type=str,
        help='Path to the file of test option, used to create metrics, etc.'
    )
    parser.add_argument(
        '--model-output-file', '--net-output-file', required=True, type=str,
        help='Path to the file of model output'
    )
    parser.add_argument(
        '--output-path', type=str,
        help='Path to save the result, default: the folder of MODEL_OUTPUT_FILE'
    )
    parser.add_argument(
        '--output-csv', type=str,
        help='Path of csv file to save the metric result'
    )
    parser.add_argument(
        '--dataset-path', type=str,
        help='Path to the dataset folder, used to override the dataset path in '
             'the TASK_OPTION_FILE'
    )
    parser.add_argument(
        '--use-gpu', default=0, type=int,
        help='GPU device to use, default 0'
    )

    args = parser.parse_args(sys.argv[1:])
    args = Args(**vars(args))
    if args.output_path is None:
        args.output_path = os.path.dirname(args.model_output_file)
    set_cuda_visible_devices([args.use_gpu])
    args.use_gpu = 0

    from settings.register_func import register_func

    register_func()

    metric_runner = MetricRunner(args)

    import torch
    with torch.no_grad():
        summary = metric_runner.run_metrics()
    if args.output_csv:
        append = os.path.exists(args.output_csv)
        save_dict_as_csv(args.output_csv, summary, append)


if __name__ == '__main__':
    main()
