from abc import ABCMeta
from functools import lru_cache

import h5py
import numpy as np
from torch.utils.data.dataset import Dataset as TorchDataset
from torchvision.transforms import Compose, ToTensor

from .torch_transforms import TensorResize
from utils import coerce_to_path_and_check_exist, use_seed
from utils.path import DATASETS_PATH

INPUT_EXTENSIONS = ['jpeg', 'jpg', 'JPG', 'png']
HDF5_FILE = 'Folsom_2014_data.hdf5'


def load_hdf5_file(filename, split):
    with h5py.File(filename, mode='r') as f:
        data = np.asarray(f['trainval']['images'], dtype=np.uint8) if split == 'train' else np.empty((0,))
    return data


class _CustomHDF5Dataset(TorchDataset):
    """Custom torch dataset from HDF5 files."""
    __metaclass__ = ABCMeta
    root = DATASETS_PATH
    name = NotImplementedError
    n_channels = 3

    def __init__(self, split, img_size, **kwargs):
        self.split = split
        self.data_path = coerce_to_path_and_check_exist(self.root / self.name / HDF5_FILE)
        data = load_hdf5_file(self.data_path, self.split)
        labels = [-1] * len(data)
        self.n_classes = 0
        
        self.data, self.labels = data, labels
        self.size = len(self.labels)

        if img_size is not None:
            self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
            assert len(self.img_size) == 2

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image = self.data[idx]
        # Convert BGR to RGB by reversing the last axis and creating a copy to avoid negative strides
        image_rgb = image[..., ::-1].copy()
        return self.transform(image_rgb), self.labels[idx]

    @property
    @lru_cache()
    def transform(self):
        transform = ToTensor()
        return transform


class FolsomDataset(_CustomHDF5Dataset):
    name = 'Folsom'
