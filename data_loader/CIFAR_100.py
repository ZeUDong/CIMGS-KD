import os
import numpy as np
from torch.utils.data import DataLoader, Sampler
from torchvision import datasets, transforms
from PIL import Image
import random


class CIFAR100Instance(datasets.CIFAR100):
    """
    CIFAR100Instance+Sample Dataset
    """
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)

        self.num_classes = 100
        num_samples = len(self.data)
        label = self.targets

        # [[],[],[]], each sub-list contains all sample ids of the class
        self.cls_positive = [[] for i in range(self.num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(self.num_classes)]
        self.cls_positive = np.asarray(self.cls_positive)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class PairBatchSampler(Sampler):
    def __init__(self, dataset, args, num_iterations=None):
        self.dataset = dataset
        self.batch_size = args.batch_size // 2
        self.num_iterations = num_iterations
        self.args = args

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        for k in range(len(self)):
            if self.num_iterations is None:
                offset = k*self.batch_size
                batch_indices = indices[offset:offset+self.batch_size]
            else:
                batch_indices = random.sample(range(len(self.dataset)),
                                              self.batch_size)

            pair_indices = []
            for idx in batch_indices:
                y = self.dataset.targets[idx]
                pair_indices.extend(list(np.random.choice(self.dataset.cls_positive[y],
                                                          1)))

            yield batch_indices + pair_indices

    def __len__(self):
        if self.num_iterations is None:
            return (len(self.dataset)+self.batch_size-1) // self.batch_size
        else:
            return self.num_iterations


def get_cifar100_dataloaders(data_folder, args=None):
    """
    cifar 100
    """
    augmentation = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4867, 0.4408],
                             [0.2675, 0.2565, 0.2761])]

    class TwoCropsTransform:
        """Take two random crops of one image as the query and key."""

        def __init__(self, base_transform):
            self.base_transform = base_transform

        def __call__(self, x):
            q = self.base_transform(x)
            k = self.base_transform(x)
            w = self.base_transform(x)
            z = self.base_transform(x)
            return [q, k, w, z]

    transforms_x = transforms.Compose(augmentation)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])


    train_set = CIFAR100Instance(root=data_folder,
                                            download=True,
                                            train=True,
                                            transform=transforms_x)

    if args.vanilla_cls:
        train_loader = DataLoader(train_set,
                                batch_sampler=PairBatchSampler(train_set, args),
                                num_workers=args.num_workers)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                pin_memory= True, num_workers=args.num_workers)

    test_set = datasets.CIFAR100(root=data_folder,
                                 download=True,
                                 train=False,
                                 transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=int(args.batch_size/2),
                             shuffle=False,
                             num_workers=int(args.num_workers/2))

    return train_loader, test_loader