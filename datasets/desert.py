import os
import sys

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2

import ai8x


class Desert(Dataset):
    labels = ["human", "no_human"]
    label_to_id_map = {k: v for v, k in enumerate(labels)}
    label_to_folder_map = {"human": "with human", "no_human": "without human"}

    def __init__(
        self,
        root_dir,
        d_type,
        transform=None,
        resize_size=(224, 224),
        augment_data=False,
    ):
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, d_type)
        if not self.__check_desert_data_exist():
            self.__print_download_manual()
            sys.exit("Dataset not found!")

        self.__get_image_paths()

        self.transform = transforms.Compose(
            [
                transform,
                transforms.Resize(resize_size),
            ]
        )

    def __check_desert_data_exist(self):
        return os.path.isdir(self.data_dir)

    def __print_download_manual(self):
        print("******************************************")
        print("      make sure to set correct path       ")
        print("******************************************")

    def __get_image_paths(self):
        self.data_list = []

        for label in self.labels:
            image_dir = os.path.join(self.data_dir, self.label_to_folder_map[label])
            for file_name in sorted(os.listdir(image_dir)):
                file_path = os.path.join(image_dir, file_name)
                if os.path.isfile(file_path) and file_name.endswith(".jpg"):
                    self.data_list.append((file_path, self.label_to_id_map[label]))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        label = torch.tensor(self.data_list[index][1], dtype=torch.int64)

        image_path = self.data_list[index][0]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        return image, label


def get_desert_15_dataset(data, load_train, load_test):
    (data_dir, args) = data

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            ai8x.normalize(args=args),
        ]
    )
    root_dir = os.path.join(data_dir, "desert_15")
    train_dataset = None
    test_dataset = None
    if load_train:
        train_dataset = Desert(
            root_dir=root_dir,
            d_type="train",
            transform=transform,
        )

    if load_test:
        test_dataset = Desert(root_dir=root_dir, d_type="test", transform=transform)

    return train_dataset, test_dataset


def get_desert_25_dataset(data, load_train, load_test):
    (data_dir, args) = data

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            ai8x.normalize(args=args),
        ]
    )
    root_dir = os.path.join(data_dir, "desert_25")
    train_dataset = None
    test_dataset = None
    if load_train:
        train_dataset = Desert(
            root_dir=root_dir,
            d_type="train",
            transform=transform,
        )

    if load_test:
        test_dataset = Desert(root_dir=root_dir, d_type="test", transform=transform)

    return train_dataset, test_dataset


def get_desert_35_dataset(data, load_train, load_test):
    (data_dir, args) = data

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            ai8x.normalize(args=args),
        ]
    )
    root_dir = os.path.join(data_dir, "desert_35")
    train_dataset = None
    test_dataset = None
    if load_train:
        train_dataset = Desert(
            root_dir=root_dir,
            d_type="train",
            transform=transform,
        )

    if load_test:
        test_dataset = Desert(root_dir=root_dir, d_type="test", transform=transform)

    return train_dataset, test_dataset


def get_desert_45_dataset(data, load_train, load_test):
    (data_dir, args) = data

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            ai8x.normalize(args=args),
        ]
    )
    root_dir = os.path.join(data_dir, "desert_45")
    train_dataset = None
    test_dataset = None
    if load_train:
        train_dataset = Desert(
            root_dir=root_dir,
            d_type="train",
            transform=transform,
        )

    if load_test:
        test_dataset = Desert(root_dir=root_dir, d_type="test", transform=transform)

    return train_dataset, test_dataset


def get_desert_55_dataset(data, load_train, load_test):
    (data_dir, args) = data

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            ai8x.normalize(args=args),
        ]
    )
    root_dir = os.path.join(data_dir, "desert_55")
    train_dataset = None
    test_dataset = None
    if load_train:
        train_dataset = Desert(
            root_dir=root_dir,
            d_type="train",
            transform=transform,
        )

    if load_test:
        test_dataset = Desert(root_dir=root_dir, d_type="test", transform=transform)

    return train_dataset, test_dataset


def get_desert_65_dataset(data, load_train, load_test):
    (data_dir, args) = data

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            ai8x.normalize(args=args),
        ]
    )
    root_dir = os.path.join(data_dir, "desert_65")
    train_dataset = None
    test_dataset = None
    if load_train:
        train_dataset = Desert(
            root_dir=root_dir,
            d_type="train",
            transform=transform,
        )

    if load_test:
        test_dataset = Desert(root_dir=root_dir, d_type="test", transform=transform)

    return train_dataset, test_dataset

def get_desert_95_dataset(data, load_train, load_test):
    (data_dir, args) = data

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            ai8x.normalize(args=args),
        ]
    )
    root_dir = os.path.join(data_dir, "desert_95")
    train_dataset = None
    test_dataset = None
    if load_train:
        train_dataset = Desert(
            root_dir=root_dir,
            d_type="train",
            transform=transform,
        )

    if load_test:
        test_dataset = Desert(root_dir=root_dir, d_type="test", transform=transform)

    return train_dataset, test_dataset

datasets = [
    {
        "name": "desert_15",
        "input": (3, 224, 224),
        "output": ("human", "no_human"),
        "loader": get_desert_15_dataset,
    },
    {
        "name": "desert_25",
        "input": (3, 224, 224),
        "output": ("human", "no_human"),
        "loader": get_desert_25_dataset,
    },
    {
        "name": "desert_35",
        "input": (3, 224, 224),
        "output": ("human", "no_human"),
        "loader": get_desert_35_dataset,
    },
    {
        "name": "desert_45",
        "input": (3, 224, 224),
        "output": ("human", "no_human"),
        "loader": get_desert_45_dataset,
    },
    {
        "name": "desert_55",
        "input": (3, 224, 224),
        "output": ("human", "no_human"),
        "loader": get_desert_55_dataset,
    },
    {
        "name": "desert_65",
        "input": (3, 224, 224),
        "output": ("human", "no_human"),
        "loader": get_desert_65_dataset,
    },
    {
        "name": "desert_95",
        "input": (3, 224, 224),
        "output": ("human", "no_human"),
        "loader": get_desert_95_dataset,
    },
]
