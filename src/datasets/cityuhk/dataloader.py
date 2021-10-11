import os

import numpy as np
from pytorch_helper.data.dataloader import DataLoaderGenerator
from pytorch_helper.settings.spaces import Spaces
from .dataset import CityUHKBEV
from .load import load_datalist


@Spaces.register(Spaces.NAME.DATALOADER, 'CityUHKBEVLoaders')
class CityUHKBEVLoaders(DataLoaderGenerator):

    def __init__(
        self, root, keys, scene_mixed, valid_ratio, use_augment, batch_size,
        num_workers, pin_memory, use_ddp
    ):
        super(CityUHKBEVLoaders, self).__init__(
            batch_size, num_workers, pin_memory, use_ddp
        )

        assert os.path.exists(root), f'{root} does not exist'
        assert os.path.isdir(root), f'{root} is not a folder'

        if isinstance(keys, str):
            self.keys = keys.strip().split(',')
        else:
            self.keys = list(keys)
        self.scene_mixed = scene_mixed
        self.valid_ratio = valid_ratio
        self.use_augment = use_augment

        self.root = root

        datalist = load_datalist(root, scene_mixed)

        self.train_key = 'train'
        self.test_key = 'test'
        self.all_key = 'all'

        num_train = len(datalist[self.train_key]) * (1 - self.valid_ratio)
        num_train = int(num_train)
        np.random.seed(0)
        np.random.shuffle(datalist[self.train_key])

        self.train_datalist = datalist[self.train_key][:num_train]
        self.valid_datalist = datalist[self.train_key][num_train:]
        self.test_datalist = datalist[self.test_key]
        self.all_datalist = datalist[self.all_key]

    def build_train_set(self):
        return CityUHKBEV(
            self.root, self.train_datalist, self.keys, self.use_augment
        )

    def build_valid_set(self):
        return CityUHKBEV(
            self.root, self.valid_datalist, self.keys, use_augment=False
        )

    def build_test_set(self):
        return CityUHKBEV(
            self.root, self.test_datalist, self.keys, use_augment=False
        )

    def build_all_set(self):
        return CityUHKBEV(
            self.root, self.all_datalist, self.keys, use_augment=False
        )

    @property
    def normalize(self):
        return CityUHKBEV.normalize

    @property
    def denormalize(self):
        return CityUHKBEV.denormalize
