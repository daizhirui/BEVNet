from typing import Optional
from typing import Tuple

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler

__all__ = ['DataLoaderGenerator']


class DataLoaderGenerator:
    STAGE_TRAIN = 'train'
    STAGE_VALID = 'valid'
    STAGE_TEST = 'test'
    STAGE_ALL = 'all'

    def __init__(
        self, batch_size: int, num_workers: int, pin_memory: bool,
        use_ddp: bool
    ):
        """ DataLoaderGenerator is an abstract class designed for single-gpu and
        multi-gpu training, validation and testing.

        :param batch_size: the number of samples in a mini-batch
        :param num_workers: the number of processes to load data
        :param pin_memory: whether to use pinned data transfer for better
            performance, see
        https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
            for more details
        :param use_ddp: whether to use `nn.parallel.DistributedDataParallel`
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_ddp = use_ddp

        self.cur_stage = None
        self._train_loader = None
        self._valid_loader = None
        self._test_loader = None
        self._all_loader = None

        self.train_sampler = None
        self.valid_sampler = None
        self.test_sampler = None
        self.all_sampler = None

    def set_epoch(self, epoch: int):
        """ This should be called at the beginning of every epoch to set the
        samplers properly such that data are sampled in an expected manner.

        :param epoch: the current epoch
        :return:
        """
        if not self.use_ddp:
            return
        if self.train_sampler:
            self.train_sampler.set_epoch(epoch)
        if self.valid_sampler:
            self.valid_sampler.set_epoch(epoch)
        if self.test_sampler:
            self.test_sampler.set_epoch(epoch)
        if self.all_sampler:
            self.all_sampler.set_epoch(epoch)

    def build_train_set(self):
        raise NotImplementedError

    def build_valid_set(self):
        raise NotImplementedError

    def build_test_set(self):
        raise NotImplementedError

    def build_all_set(self):
        raise NotImplementedError

    def build_dataloader(
        self, dataset: Dataset, shuffle: bool, collate_fn=None, **kwargs
    ) -> Tuple[DataLoader, Optional[DistributedSampler]]:
        """ This method builds the dataloader and the distributed sampler if
        `self.use_ddp` is True.

        :param dataset: the Dataset instance
        :param shuffle: whether to randomly sample the dataset
        :param collate_fn: callable to collate a mini-batch
        :return: the dataloader and the sampler
        """
        sampler = None
        if self.use_ddp:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = shuffle and sampler is None
        kw = dict(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            collate_fn=collate_fn
        )
        kw.update(kwargs)
        dataloader = DataLoader(**kw)
        return dataloader, sampler

    @property
    def train_loader(self):
        """ build and return the dataloader for training

        :return: the dataloader for training
        """
        self.cur_stage = self.STAGE_TRAIN
        if self._train_loader is None:
            self._train_loader, self.train_sampler = self.build_dataloader(
                self.build_train_set(), shuffle=True
            )

        return self._train_loader

    @property
    def valid_loader(self):
        """ build and return the dataloader for validation

        :return: the dataloader for validation
        """
        self.cur_stage = self.STAGE_VALID
        if self._valid_loader is None:
            self._valid_loader, self.valid_sampler = self.build_dataloader(
                self.build_valid_set(), shuffle=False
            )

        return self._valid_loader

    @property
    def test_loader(self):
        """ build and return the dataloader for testing

        :return: the dataloader for testing
        """
        self.cur_stage = self.STAGE_TEST
        if self._test_loader is None:
            self._test_loader, self.test_sampler = self.build_dataloader(
                self.build_test_set(), shuffle=False
            )

        return self._test_loader

    @property
    def all_loader(self):
        """ build and return the dataloader that will load all the data, for
        training, validation and testing

        :return: the dataloader for the whole dataset
        """
        self.cur_stage = self.STAGE_ALL
        if self._all_loader is None:
            self._all_loader, self.all_sampler = self.build_dataloader(
                self.build_all_set(), shuffle=False
            )

        return self._all_loader
