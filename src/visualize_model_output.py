import argparse
import multiprocessing as mp
import os.path

import h5py
import numpy as np
from tqdm import tqdm

from pytorch_helper.utils.log import get_logger


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-output-file', required=True, type=str,
        help='Path of the model output file.'
    )
    parser.add_argument(
        '--dataset-path',
        default=os.path.join(os.curdir, 'data', 'CityUHK-X-BEV'), type=str,
        help='Path of the dataset.'
    )
    parser.add_argument(
        '--use-gpu', default=0, type=int, help='Index of the gpu to use.'
    )
    parser.add_argument(
        '-j', default=1, type=int, help='Number of workers.'
    )
    return parser.parse_args()


def worker(args, queue_in, queue_out):
    from pytorch_helper.utils.pre_pytorch_init import set_cuda_visible_devices
    set_cuda_visible_devices([args.use_gpu])

    import torch
    import torch.nn.modules as nn
    from torchvision.transforms.functional import to_pil_image
    from pytorch_helper.utils.image import to_heatmap
    from pytorch_helper.utils.io.image import imsave
    from pytorch_helper.utils.io import config

    from models.bevnet import BEVTransform
    from settings.register_func import register_func
    from tasks.helper import assemble_3in1_image
    from utils.visualize import merge_head_map_and_feet_map
    from utils.visualize import overlay_heatmap_on_image
    from utils.visualize import save_density_map

    config.silent = True
    register_func()

    model_output = h5py.File(args.model_output_file, 'r')

    output_dir = os.path.dirname(args.model_output_file)
    output_dir = os.path.join(output_dir, 'visualize')

    bev_transform = BEVTransform()
    loss_fn = nn.MSELoss()

    def get_path(name):
        return os.path.join(output_dir, name)

    target_files = []

    def save_image(func, idx, file_path, im):
        func(file_path, im)
        target_files.append(f'{file_path}.{config.img_ext}')
        queue_out.put((idx, file_path, os.getpid()))

    scene_ids = model_output['scene_id'][:]
    image_ids = model_output['image_id'][:]

    while True:
        try:
            data = queue_in.get(block=True)

            if isinstance(data, str) and data == 'exit':
                queue_out.put(os.getpid())
                exit()

            index = data
            # index = 0
            scene_id = int(scene_ids[index])
            image_id = int(image_ids[index])

            h5file = f'scene_{scene_id:03d}.h5'
            h5file = h5py.File(os.path.join(args.dataset_path, h5file), 'r')

            prefix = f'scene{scene_id:02d}-{image_id}'

            # save input
            path = get_path(f'{prefix}-input')
            input_im = h5file['image'][image_id]
            input_im_tensor = torch.tensor(input_im)
            input_im = input_im.transpose(1, 2, 0)
            imsave(path, input_im)

            # get maps
            pred_bev_map = model_output['bev_map'][index, 0]
            gt_bev_map = h5file['bev_map'][image_id, 0]

            gt_head_map = h5file['head_map'][image_id, 0]
            if 'head_map' in model_output:
                pred_head_map = model_output['head_map'][index, 0]
            else:
                pred_head_map = None

            gt_feet_map = h5file['feet_map'][image_id, 0]
            if 'feet_map' in model_output:
                pred_feet_map = model_output['feet_map'][index, 0]
            else:
                pred_feet_map = None

            # get pose and loss
            pred_height = model_output['camera_height'][index].item()
            gt_height = h5file['camera_height'][image_id].item()
            pred_angle = model_output['camera_angle'][index].item()
            gt_angle = h5file['camera_angle'][image_id].item()
            pose_height_loss = loss_fn(
                torch.tensor(pred_height), torch.tensor(gt_height)
            ).item()
            pose_angle_loss = loss_fn(
                torch.tensor(pred_angle), torch.tensor(gt_angle)
            ).item()

            # get camera fu, fv
            camera_fu = h5file['camera_fu'][image_id]
            camera_fv = h5file['camera_fv'][image_id]

            # save 3in1 image: input-gt-pred
            for k, pred_map, gt_map in [
                ('head', pred_head_map, gt_head_map),
                ('feet', pred_feet_map, gt_feet_map),
                ('bev', pred_bev_map, gt_bev_map)
            ]:
                if pred_map is None:
                    continue

                # save the map
                path = get_path(f'{prefix}-{k}-gt')
                save_image(save_density_map, index, path, gt_map)
                path = get_path(f'{prefix}-{k}-pred')
                save_image(save_density_map, index, path, pred_map)

                map_loss = loss_fn(
                    torch.tensor(pred_map), torch.tensor(gt_map)
                ).item()

                titles = [
                    # input
                    f'input, map loss={map_loss:.2e}\n'
                    f'height loss={pose_height_loss:.2e}, '
                    f'angle loss={pose_angle_loss:.2e}',
                    # gt
                    f'g.t. camera height={gt_height:.2e}m\n'
                    f'camera angle={gt_angle:.2e}',
                    # pred
                    f'p.d. camera height={pred_height:.2e}m\n'
                    f'camera angle={pred_angle:.2e}'
                ]
                image = assemble_3in1_image(
                    input_im_tensor, torch.tensor(gt_map),
                    torch.tensor(pred_map), titles
                )
                path = get_path(f'{prefix}-3in1-{k}')
                save_image(imsave, index, path, to_pil_image(image))

            # project detection of head and feet on to images, synthesize and
            # save the result
            for j, bev_map, head_map, feet_map, camera_height, camera_angle in [
                ('pred', pred_bev_map, pred_head_map, pred_feet_map,
                 pred_height, pred_angle),
                ('gt', gt_bev_map, gt_head_map, gt_feet_map,
                 gt_height, gt_angle)
            ]:
                h, w = bev_map.shape
                # save bev back-projection
                iv_roi = bev_transform.get_iv_roi(
                    [1, 1, h, w], torch.tensor([camera_height]),
                    torch.tensor([camera_angle]), camera_fu, camera_fv
                )[0, 0].cpu().numpy()

                bev_back = bev_transform(
                    torch.tensor(bev_map[np.newaxis, np.newaxis]), 0,
                    torch.tensor([camera_height]), torch.tensor([camera_angle]),
                    camera_fu,
                    camera_fv,
                    i2b=False
                )[0][0, 0].cpu().numpy()
                image = overlay_heatmap_on_image(
                    input_im / 255, to_heatmap(bev_back), bev_back
                )
                image[iv_roi, 0] = 1
                path = get_path(f'{prefix}-bev_back-{j}')
                save_image(imsave, index, path, image)

                # save feet and head detection as a synthesized image
                heatmap, weight_map = merge_head_map_and_feet_map(
                    head_map, feet_map
                )
                image = overlay_heatmap_on_image(
                    input_im / 255, heatmap, weight_map,
                )

                image[iv_roi, 0] = 1.0
                image = (image * 255).astype(np.uint8)
                path = get_path(f'{prefix}-output-{j}-syn')
                save_image(imsave, index, path, image)

        except Exception as e:
            queue_out.put(e)


def main():
    args = parse()
    assert args.j >= 1, 'Require at least one worker.'
    queue_in = mp.Queue()
    queue_out = mp.Queue()
    logger = get_logger(__name__)
    pool = []
    pool_ack = []
    try:
        for _ in range(args.j):
            p = mp.Process(target=worker, args=(args, queue_in, queue_out))
            p.start()
            pool.append(p)

        model_input = h5py.File(args.model_output_file)
        size = model_input['scene_id'].size
        for i in range(size):
            queue_in.put(i)
        for _ in range(args.j):
            queue_in.put('exit')

        processed_index = set()
        target_files = set()
        with tqdm(total=size, ncols=80) as bar:
            while len(processed_index) < size:
                data = queue_out.get(block=True)
                if isinstance(data, Exception):
                    raise data  # exception
                if isinstance(data, int):
                    # the subprocess reports its pid to exit
                    pool_ack.append(data)
                else:
                    if data[0] not in processed_index:
                        bar.update()
                    if data[1] in target_files:
                        raise ValueError(f'Repeated file: {data}')
                    processed_index.add(data[0])
                    target_files.add(data[1])
                    logger.info(f'{len(target_files)}, new: {data}')

            # waiting for all workers to exit
            while len(pool_ack) < len(pool):
                data = queue_out.get(block=True)
                if isinstance(data, Exception):
                    raise data  # exception
                if isinstance(data, int):
                    # the subprocess reports its pid to exit
                    pool_ack.append(data)
                else:
                    if data[1] in target_files:
                        raise ValueError(f'Repeated file: {data}')
                    processed_index.add(data[0])
                    target_files.add(data[1])
                    logger.info(f'{len(target_files)}, new: {data}')
    finally:
        for p in pool:
            p.terminate()
            p.join()
            logger.info(f'Process {p.pid} joined.')

        queue_in.close()
        queue_out.close()


if __name__ == '__main__':
    main()
