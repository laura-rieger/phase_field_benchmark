import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BasicDataset(Dataset):

    def __init__(self,
                 images_dir: str,
                 offset=50,
                 start_offset=10,
                 scale: float = 1.0,
                 mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.offset = offset
        self.start_offset = start_offset
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [
            splitext(file)[0] for file in listdir(images_dir)
            if not file.startswith('.')
        ]

        if not self.ids:
            raise RuntimeError(
                f'No input file found in {images_dir}, make sure you put your images there'
            )
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids) - self.offset - self.start_offset

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize(
            (newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)
        img_ndarray = img_ndarray > img_ndarray.mean()

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            # img_ndarray = img_ndarray / 255
            # img_ndarray = (img_ndarray - .5) * 2

        return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext == '.npy':
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        idx = idx + self.start_offset  # XXX maybe confusing?

        name = self.ids[idx]
        mask_name = self.ids[idx + self.offset]

        mask_file = list(self.images_dir.glob(mask_name + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(
            img_file
        ) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(
            mask_file
        ) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)[0]
        mask = self.preprocess(mask, self.scale, is_mask=True)
        return torch.as_tensor(
            img.copy()).float().contiguous(), torch.as_tensor(
                mask.copy()).float().contiguous()


class RamDataset(BasicDataset):

    def __init__(self,
                 images_file: str,
                 offset=50,
                 start_offset=10,
                 scale: float = 1.0,
                 mask_suffix: str = '',
                num_channels =1):
        self.images_file = images_file
        self.data_arr = np.load(images_file).reshape(-1, 100, 100)

        self.data_arr = self.data_arr > self.data_arr.mean()
        self.num_channels = num_channels

        self.offset = offset
        self.start_offset = start_offset
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        logging.info(f'Creating dataset with {len(self.data_arr)} examples')

    def __len__(self):
        return np.maximum(
            len(self.data_arr) - self.offset - self.start_offset, 0)

    def __getitem__(self, idx):
        idx = idx + self.start_offset  

        img = self.preprocess(self.data_arr[idx - self.num_channels:idx], self.scale, is_mask=False)[0]
        mask = self.preprocess(self.data_arr[idx + self.offset],
                               self.scale,
                               is_mask=True)
        return torch.as_tensor(
            img, dtype=torch.float).contiguous(), torch.as_tensor(
                mask, dtype=torch.float).contiguous()

    @staticmethod
    def preprocess(img_ndarray, scale, is_mask):

        # img_ndarray = img_ndarray > img_ndarray.mean()

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]


        return img_ndarray
import h5py

from torch.utils.data import Dataset

class H5pyDataset(Dataset):
    def __init__(self, h5_file_path, transform=None,                 
                offset=50,
                 start_offset=10,
                 scale: float = 1.0,
                 mask_suffix: str = '',
                num_channels =1, 
                percentage = 1.0):
        """
        Args:
            h5_file_path (str): Path to the HDF5 file.
            transform (callable, optional): Optional transform to be applied to the loaded data.
        """
        self.h5_file_path = h5_file_path
        self.transform = transform
        self.offset = offset
        self.start_offset = start_offset
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.num_channels = num_channels
        
        # Open the HDF5 file for reading
        self.h5file = h5py.File(h5_file_path, 'r')
        self.key_list = list(self.h5file.keys())
        vals_to_be_removed = [] # too short trajectory
        for value in self.key_list:
            if self.h5file[value].shape[0] <= self.offset + self.start_offset:
                vals_to_be_removed.append(value)
        for value in vals_to_be_removed:
            self.key_list.remove(value)
        # shuffle list
        np.random.seed(0)

        np.random.shuffle(self.key_list)
        # cut off list
        num_used = int(len(self.key_list) * percentage)
        self.key_list = self.key_list[:num_used]
        # print(self.key_list)
      
        self.num_samples = 0
        self.num_list = []
        for key in self.key_list:
            self.num_samples += self.h5file[key].shape[0] - offset - start_offset
            self.num_list.append(self.h5file[key].shape[0] - offset - start_offset)
        
        self.cum_num_list = np.cumsum(self.num_list)
    def get_num_traj(self):
        return len(self.key_list)
    def get_len_traj(self, traj_idx):
        return self.h5file[self.key_list[traj_idx]].shape[0] - self.offset - self.start_offset
    def get_trajectory(self, indicator, index):
        img = self.preprocess(self.h5file[self.key_list[indicator]][self.start_offset + index], 
                                self.scale, is_mask=False)
        img = torch.as_tensor(
            img, dtype=torch.float).contiguous()
        if self.transform:
            img = self.transform(img)
        return img
        # return torch.as_tensor(
        #     img, dtype=torch.float).contiguous()




    @staticmethod
    def preprocess(img_ndarray, scale, is_mask):

        # img_ndarray = img_ndarray > img_ndarray.mean()

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis,  ...]
        # img_ndarray = img_ndarray > img_ndarray.mean()
        # xxx

        return img_ndarray
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        if idx >= self.num_samples:
            raise IndexError
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # identify which dataset
        dataset_idx = np.argmax(idx < self.cum_num_list)
        # print(self.key_list[dataset_idx])
        # identify the index in the dataset
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cum_num_list[dataset_idx - 1]
        # get the data
        sample = self.h5file[self.key_list[dataset_idx]][sample_idx+ self.start_offset]
        sample_end = self.h5file[self.key_list[dataset_idx]][sample_idx + self.offset+ self.start_offset]
        img = self.preprocess(sample, self.scale, is_mask=False)
        mask = self.preprocess(sample_end, self.scale, is_mask=True)
        img = torch.as_tensor(
            img, dtype=torch.float).contiguous()
        if self.transform:
            img = self.transform(img)
        return img, torch.as_tensor(
                mask, dtype=torch.float).contiguous()

