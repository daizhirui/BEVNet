import argparse
import os
import sys
from collections import namedtuple
from dataclasses import dataclass
from typing import Dict
from typing import List

import h5py
import numpy as np
import pandas
import torch
from skimage.transform import resize

dir_name = os.path.dirname(__file__)
sys.path.append(os.path.realpath(os.path.join(dir_name, '..', '..')))

from models.bevnet.bev_transform import BEVTransform
from datasets.cityuhk.kernels import GaussianKernel
from pytorch_helper.utils.log import get_logger, pbar

AnnotationItem = namedtuple(
    'AnnotationItem', ['url', 'source', 'head', 'feet', 'type']
)

logger = get_logger(__name__)


def load_annotation(file_path: str) -> Dict[str, List[AnnotationItem]]:
    """ load_annotation loads annotations from file_path and parses them into a
    dictionary of annotations with image sources as keys.
    """
    if file_path.endswith('.csv'):
        file = pandas.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        file = pandas.read_excel(file_path)
    else:
        logger.error('Unknown file type')
        return dict()
    annotations: Dict[str, List] = dict()
    n_rows = file.shape[0]
    for r in range(n_rows):
        annotations.setdefault(file['source'][r], []).append(
            AnnotationItem(
                url=file['url'][r],
                source=file['source'][r],
                head=eval(file['head'][r]),
                feet=eval(file['feet'][r]),
                type=file['type'][r]
            )
        )
    return annotations


def generate_h5files_for_scenes(config):
    if not os.path.exists(config.output_path):
        os.mkdir(config.output_path)
    out_height, out_width = config.output_size
    ori_height, ori_width = config.original_size
    # prepare transformer and gaussian kernel
    device = torch.device(f'cuda:{config.use_device}')

    def to_tensor(x):
        return torch.tensor(x, dtype=torch.float32).to(device)

    transformer: BEVTransform = BEVTransform()
    sigma_default = config.gaussian_sigma
    kernel_size = 6 * sigma_default + 1
    gaussian_kernel = GaussianKernel(kernel_size, sigma_default, channels=1)
    gaussian_kernel.to(device)

    p_bar = pbar(range(1, 56), position=1)
    for scene_id in p_bar:
        # create file handlers
        scene_name = f'scene_{scene_id:03d}'
        h5_name = f'{scene_name}.h5'
        input_h5_file_path = os.path.join(config.input_path, h5_name)
        output_h5_file_path = os.path.join(config.output_path, h5_name)
        annotation_file_path = os.path.join(
            config.annotation_folder, scene_name, 'annotations.csv'
        )
        if not os.path.exists(annotation_file_path):
            logger.warn(
                f"annotation for scene {scene_id} doesn't exist, skip it.")
            p_bar.update()
            continue
        input_h5 = h5py.File(input_h5_file_path, 'r')
        output_h5 = h5py.File(output_h5_file_path, 'w')

        # load annotations and original image filenames of this scene
        annotations = load_annotation(annotation_file_path)
        n_images = input_h5['filenames'].shape[0]
        original_filenames = [
            input_h5['filenames'][i].decode('utf-8') for i in range(n_images)
        ]
        if len(annotations) < n_images:
            logger.error(f'Abort generating scene {scene_id} due to the lack '
                         f'of annotations')
            logger.error('The following images have no annotations')
            for name in original_filenames:
                if name not in annotations:
                    logger.error(name)
            output_h5.close()
            os.remove(output_h5_file_path)
            input_h5.close()
            exit(-1)

        logger.info(f"Make sub-dataset 'image' for scene {scene_id}")
        input_images_data = input_h5['images']
        in_height, in_width = input_images_data.shape[2:]
        down_scale_factor = in_height / out_height
        assert down_scale_factor == in_width / out_width, \
            'Aspect ratios of input and output are different'

        if down_scale_factor == 1:
            images_data = input_images_data[...]
        else:
            images_data = np.empty(
                (n_images, 3, out_height, out_width),
                dtype=input_images_data.dtype
            )
            for i in range(n_images):
                resized_im = resize(
                    image=np.transpose(input_images_data[i], [1, 2, 0]),
                    output_shape=config.output_size, anti_aliasing=True
                )
                images_data[i] = np.transpose(resized_im, [2, 0, 1])

        logger.info('Calculate focal length')
        fov_h = input_h5.get('camera_horizontal_fov', [57.7])[0]
        fov_v = input_h5.get('camera_vertical_fov', [44.6])[0]
        fu = ori_width / (2 * np.tan(fov_h * np.pi / 360))
        fv = ori_height / (2 * np.tan(fov_v * np.pi / 360))
        camera_fu = to_tensor(fu).repeat(n_images)
        camera_fv = to_tensor(fv).repeat(n_images)
        camera_height = to_tensor(input_h5['camera_height'][:])
        camera_angle = to_tensor(-input_h5['camera_angle'][:] / 180. * np.pi)

        logger.info('Collect head & feet annotation')
        n_annotations = np.zeros(n_images, dtype=np.uint8)
        # max_n_annotations = max([len(v) for v in annotations.values()])
        max_n_annotations = 121
        shape = (n_images, 3, max_n_annotations)
        feet_annotations = torch.zeros(shape).to(device)
        head_annotations = torch.zeros(shape).to(device)
        feet_annotations[:, 2, :] = 1
        head_annotations[:, 2, :] = 1
        generated_record = np.zeros(n_images)
        for image_source, image_annotations in annotations.items():
            try:
                index = original_filenames.index(image_source)
                generated_record[index] = 1
                n_annotations[index] = len(image_annotations)
                for ann_index, ann in enumerate(image_annotations):
                    feet_annotations[index, 0, ann_index] = ann.feet[1]
                    feet_annotations[index, 1, ann_index] = ann.feet[0]
                    head_annotations[index, 0, ann_index] = ann.head[1]
                    head_annotations[index, 1, ann_index] = ann.head[0]
            except ValueError as e:
                logger.error(str(e))
                p_bar.update()
                continue
        if generated_record.sum() != generated_record.size:
            logger.warn(f'Some images have no annotations!\n'
                        f'Image index: {np.where(generated_record != -1)[0]}')

        logger.info('Calculate feet positions on the ground plane')
        homo_inv_mats, _, _ = transformer.get_bev_param(
            config.original_size, camera_height, camera_angle, camera_fu,
            camera_fv, w2i=False
        )
        feet_positions = transformer.image_coord_to_world_coord(
            feet_annotations, homo_inv_mats
        )

        logger.info('Adjust annotation and camera parameters')
        s = out_height / ori_height
        # annotation is now under the scale of the output size
        feet_annotations[:, :2, :] *= s
        head_annotations[:, :2, :] *= s
        camera_fu *= s
        camera_fv *= s

        logger.info('Calculate feet positions in BEV frame')
        homo_inv_mats, scales, centers = transformer.get_bev_param(
            config.output_size, camera_height, camera_angle, camera_fu,
            camera_fv, w2i=False
        )
        feet_bev_pixels = transformer.world_coord_to_bev_coord(
            config.output_size, feet_positions, scales, centers
        )

        map_shape = (n_images, 1, out_height, out_width)
        logger.info('Generate feet and head maps in image frame')
        feet_maps = torch.zeros(map_shape).to(device)
        head_maps = torch.zeros(map_shape).to(device)
        for i in range(n_images):
            for j in range(n_annotations[i]):
                u_f, v_f = feet_annotations[i, :2, j].cpu().long()
                u_h, v_h = head_annotations[i, :2, j].cpu().long()
                if 0 <= u_f < out_width and 0 <= v_f < out_height:
                    feet_maps[i, 0, v_f, u_f] = 1
                if 0 <= u_h < out_width and 0 <= v_h < out_height:
                    head_maps[i, 0, v_h, u_h] = 1
        feet_maps = gaussian_kernel(feet_maps).cpu().numpy()
        head_maps = gaussian_kernel(head_maps).cpu().numpy()

        logger.info('Generate Binary BEV maps')
        bev_maps = torch.zeros(map_shape).to(device)
        for i in range(n_images):
            for j in range(n_annotations[i]):
                u_b, v_b = feet_bev_pixels[i, :2, j].cpu().type(torch.long)
                if 0 <= u_b < out_width and 0 <= v_b < out_height:
                    bev_maps[i, 0, v_b, u_b] = 1
        bev_maps_data = gaussian_kernel(bev_maps).cpu().numpy()

        roi_mask_data = input_h5['roi_mask'].__array__()

        logger.info(f'Saving to file {output_h5_file_path}')

        key_data = {
            'image': images_data,
            'roi_mask': roi_mask_data,
            'world_coord': feet_positions.cpu().numpy(),
            'bev_coord': feet_bev_pixels.cpu().numpy(),
            'bev_center': centers.cpu().numpy(),
            'bev_scale': scales.cpu().numpy(),
            'bev_map': bev_maps_data,
            'feet_map': feet_maps,
            'head_map': head_maps,
            'camera_fu': camera_fu.cpu().numpy(),
            'camera_fv': camera_fv.cpu().numpy(),
            'camera_height': camera_height.cpu().numpy(),
            'camera_angle': camera_angle.cpu().numpy(),
            'feet_annotation': feet_annotations.cpu().numpy(),
            'head_annotation': head_annotations.cpu().numpy(),
            'num_annotations': n_annotations
        }
        for name, data in pbar(key_data.items(), position=0,
                               total=len(key_data)):
            logger.info(f'Saving dataset {name}')
            output_h5.create_dataset(
                name=name, shape=data.shape, dtype=data.dtype, data=data,
                compression='gzip'
            )
        output_h5.attrs['scene_id'] = scene_id
        output_h5.close()
        input_h5.close()
    p_bar.close()


@dataclass
class Config(object):
    output_path: str
    output_size: list
    gaussian_sigma: int
    input_path: str
    annotation_folder: str
    use_device: int
    original_size = [1536, 2048]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output-path', type=str,
        default=os.path.join(os.curdir, 'data', 'CityUHK-X-BEV'),
        help='Path of the folder to save the generated dataset.'
    )
    parser.add_argument(
        '--output-size', default=[384, 512],
        help='Size of the output image and ground truth maps.'
    )
    parser.add_argument(
        '--gaussian-sigma', type=int, default=5,
        help='Std of the gaussian kernel to generate density map'
    )
    parser.add_argument(
        '--input-path', type=str,
        default=os.path.join(os.curdir, 'data', 'CityUHK-X-Original'),
        help='Path to the folder of the original CityUHK-X dataset.'
    )
    parser.add_argument(
        '--annotation-folder', type=str,
        default=os.path.join(os.curdir, 'data', 'annotations'),
        help='Path to the annotation folder'
    )
    parser.add_argument('--use-device', type=int, default=0)
    arg = parser.parse_args(sys.argv[1:])
    config = Config(**vars(arg))
    generate_h5files_for_scenes(config)


if __name__ == '__main__':
    main()
