import argparse
import os

import h5py
import numpy as np
from pytorch_helper.utils import log
from pytorch_helper.utils.io import save_as_pickle


def create_split(output_path):
    scenes = list(range(0, 56))
    # following scenes have no BEV annotations
    del scenes[18]
    del scenes[16]
    del scenes[14]
    del scenes[13]
    del scenes[0]
    scenes = np.array(scenes, dtype=np.int32)
    np.random.seed(30)
    np.random.shuffle(scenes)
    train_scenes = scenes[:43].tolist()
    test_scenes = scenes[43:].tolist()
    train_scenes.sort()
    test_scenes.sort()

    dataset = dict(
        train_scene=train_scenes, test_scene=test_scenes,
        train=[], test=[], all=[]
    )

    for scene_id in scenes:
        h5_file = os.path.join(output_path, f'scene_{scene_id:03d}.h5')
        n_images = h5py.File(h5_file, 'r')['image'].shape[0]
        for image_id in range(n_images):
            dataset['all'].append([scene_id, image_id])

    for key, scenes in log.pbar(
        zip(['train', 'test'], [train_scenes, test_scenes])
    ):
        for scene_id in scenes:
            h5_file = os.path.join(output_path, f'scene_{scene_id:03d}.h5')
            n_images = h5py.File(h5_file, 'r')['image'].shape[0]

            for image_id in range(n_images):
                dataset[key].append([scene_id, image_id])

    save_as_pickle(os.path.join(output_path, f'scene-split.datalist'), dataset)


def create_mixed(output_path):
    test_ratio = 0.15
    scenes = list(range(0, 56))
    del scenes[18]
    del scenes[16]
    del scenes[14]
    del scenes[13]
    del scenes[0]

    images = []
    dataset = dict(all=[])

    for scene_id in log.pbar(scenes):
        scene_file = os.path.join(output_path, f'scene_{scene_id:03d}.h5')
        scene_data = h5py.File(scene_file, 'r')
        n_images = scene_data['image'].shape[0]
        for image_id in range(n_images):
            images.append([scene_id, image_id])
            dataset['all'].append([scene_id, image_id])

    num_train = int((1 - test_ratio) * len(images))
    np.random.seed(0)
    np.random.shuffle(images)
    dataset['train'] = images[:num_train]
    dataset['test'] = images[num_train:]
    save_as_pickle(os.path.join(output_path, f'scene-mixed.datalist'), dataset)


def main():
    parser = argparse.ArgumentParser('Build datalist for training and testing')
    parser.add_argument(
        '--output-path', type=str,
        default=os.path.join(os.curdir, 'data', 'CityUHK-X-BEV'),
        help='Path to the dataset folder, also the path to save datalist files'
    )
    parser.add_argument(
        '--mixed', action='store_true',
        help='Generate datalist with all scenes mixed together, otherwise the '
             'set of scenes in training set and testing set will be disjoint'
    )
    args = parser.parse_args()

    if args.mixed:
        create_mixed(args.output_path)
    else:
        create_split(args.output_path)


if __name__ == '__main__':
    main()
