import logging
import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision.datasets import FashionMNIST

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class FashionMNIST_truncated(data.Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        fashion_mnist_dataobj = FashionMNIST(self.root, self.train, self.transform, self.target_transform, self.download)

        if self.train:
            data = fashion_mnist_dataobj.data
            target = np.array(fashion_mnist_dataobj.targets)
        else:
            data = fashion_mnist_dataobj.data
            target = np.array(fashion_mnist_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        
        # Convert to PIL Image for transforms
        img = Image.fromarray(img.numpy(), mode='L')
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)