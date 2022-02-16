import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# data folder
DATA_FOLDER = "../data/"


def load_data_from_path(path):
    data = torch.load(path)
    return data["images"], data["labels"]


def get_dataloader(batch_size, train_size=None, test_size=None, transform_train_data=True):
    """
        returns: cifar dataloader

    Arguments:
        batch_size:
        train_size: How many samples to use of train dataset?
        test_size: How many samples to use from test dataset?
        transform_train_data: If we should transform (random crop/flip etc) or not
    """

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4),
            transforms.ToTensor(), normalize
        ]
    ) if transform_train_data else transforms.Compose([transforms.ToTensor(), normalize])

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    # CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root=DATA_FOLDER, train=True, transform=transform, download=True
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=DATA_FOLDER, train=False, transform=test_transform, download=True
    )

    if train_size:
        indices = np.random.permutation(np.arange(len(train_dataset)))
        train_dataset = Subset(train_dataset, indices[:train_size])

    if test_size:
        indices = np.random.permutation(np.arange(len(test_dataset)))
        test_dataset = Subset(train_dataset, indices[:test_size])

    # Data loader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, test_loader
