import logging

import numpy as np
import torch.utils.data as data
from torchvision.datasets import SVHN

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class SVHN_truncated(data.Dataset):
    def __init__(
        self,
        root,
        dataidxs=None,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        split = "train" if self.train else "test"
        svhn_dataobj = SVHN(
            self.root,
            split=split,
            transform=self.transform,
            target_transform=self.target_transform,
            download=self.download,
        )

        data_arr = svhn_dataobj.data
        target_arr = np.array(svhn_dataobj.labels)
        target_arr[target_arr == 10] = 0
        data_arr = np.transpose(data_arr, (0, 2, 3, 1))

        if self.dataidxs is not None:
            data_arr = data_arr[self.dataidxs]
            target_arr = target_arr[self.dataidxs]

        return data_arr, target_arr

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

