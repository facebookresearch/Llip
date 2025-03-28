"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import os

import numpy as np
from PIL import Image

import torch
from torchvision import datasets as t_datasets
from llip.clipeval.mit_states import MITStates


class INv2Folder(t_datasets.ImageFolder):
    def __init__(self,
                 root: str,
                 transform=None,
                 target_transform=None,
                 ):
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
        )

    def find_classes(self, directory: str):
        """Finds the class folders in a dataset.

        See :class:`DatasetFolder` for details.
        """
        classes = sorted(int(entry.name)
                         for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(
                f"Couldn't find any class folder in {directory}.")

        class_to_idx = {str(cls_name): i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def clevr_t_transform(x):
    return [
        "count_10",
        "count_3",
        "count_4",
        "count_5",
        "count_6",
        "count_7",
        "count_8",
        "count_9",
    ].index(x)


class FileListDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.images = np.load(images)
        self.labels = np.load(labels)

    def __getitem__(self, index):
        img = pil_loader(self.images[index])
        target = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.images)


def get_downstream_dataset(catalog, name, is_train, transform):
    entry = catalog[name]
    root = entry["path"]
    if entry["type"] == "imagefolder":
        if name == "imagenet_v2":
            dataset = INv2Folder(
                os.path.join(root, entry["train"]
                             if is_train else entry["test"]),
                transform=transform,
            )

        else:
            dataset = t_datasets.ImageFolder(
                os.path.join(root, entry["train"]
                             if is_train else entry["test"]),
                transform=transform,
            )
    elif entry["type"] == "special":
        if name == "CIFAR10":
            dataset = t_datasets.CIFAR10(
                root, train=is_train, transform=transform, download=True
            )
        elif name == "CIFAR100":
            dataset = t_datasets.CIFAR100(
                root, train=is_train, transform=transform, download=True
            )
        elif name == "STL10":
            dataset = t_datasets.STL10(
                root,
                split="train" if is_train else "test",
                transform=transform,
                download=True,
            )
        elif name == "MNIST":
            dataset = t_datasets.MNIST(
                root, train=is_train, transform=transform, download=True
            )
        elif name == "MITStates":
            dataset = MITStates(data_dir=root)
    elif entry["type"] == "filelist":
        path = entry["train"] if is_train else entry["test"]
        val_images = os.path.join(root, path + "_images.npy")
        val_labels = os.path.join(root, path + "_labels.npy")
        if name == "CLEVRCounts":
            target_transform = Compose([clevr_t_transform])
        else:
            target_transform = None
        dataset = FileListDataset(val_images, val_labels, transform, target_transform)
    else:
        raise Exception("Unknown dataset")

    return dataset
