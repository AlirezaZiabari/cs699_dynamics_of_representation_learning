from pkgutil import get_data
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
import os, sys

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# data folder
DATA_FOLDER = "../data/"


def load_data_from_path(path):
    data = torch.load(path)
    return data["images"], data["labels"]


def get_dataloader(batch_size, train_size=None, test_size=None, transform_train_data=True, get_adversarial=False, root_dir=None):
    """
        returns: cifar dataloader

    Arguments:
        batch_size:
        train_size: How many samples to use of train dataset?
        test_size: How many samples to use from test dataset?
        transform_train_data: If we should transform (random crop/flip etc) or not
    """

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    transform_default = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4),
            transforms.ToTensor(), 
            normalize
        ]
    ) if transform_train_data else transforms.Compose([transforms.ToTensor(), normalize])
    
    transform_adversarial = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4),
            transforms.ToTensor(), 
            normalize
        ]
        
    ) if transform_train_data else transforms.Compose([transforms.ToTensor(), normalize])

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    if get_adversarial:
        # Adversarial CIFAR-10 Dataset
        train_dataset = AdversarialDataset(root_dir=root_dir, train=True)
        test_dataset = AdversarialDataset(root_dir=root_dir, train=False)
    else:
        # CIFAR-10 dataset
        train_dataset = torchvision.datasets.CIFAR10(
            root=DATA_FOLDER, train=True, transform=transform_default, download=True
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


class AdversarialDataset(Dataset):
    
    def __init__ (self, root_dir, train=True, transform=None):
        self.transform = transform
        self.train = train
        if self.train:
            self.root_dir = os.path.join(root_dir, 'train')
        else:
            self.root_dir = os.path.join(root_dir, 'test')
        self._get_dir_list()
        
    def __len__(self):
        return len(self.dir_list)
    
    def _get_dir_list(self):
        self.dir_list = os.listdir(self.root_dir)
        self.dir_list = [i for i in self.dir_list if not (i.startswith('.') or i[-3:] == 'ini')]
    
    def __getitem__(self, index):
        
        if torch.is_tensor(index):
            idx = idx.tolist()

        data_path = os.path.join(self.root_dir, self.dir_list[index])
        data =  torch.load(data_path)
        
        image, label = data["images"].squeeze(), data["labels"].squeeze()
        if self.transform:
            image = self.transform(image.squeeze()).unsqueeze(dim=0)
            
        sample = (image, label)
        return sample

    
    
        