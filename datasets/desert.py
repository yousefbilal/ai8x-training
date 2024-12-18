import os
import sys

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as album
import cv2

import ai8x


class Desert(Dataset):
    """
    `Cats vs Dogs dataset <https://www.kaggle.com/datasets/salader/dogs-vs-cats>` Dataset.

    Args:
    root_dir (string): Root directory of dataset where ``KWS/processed/dataset.pt``
        exist.
    d_type(string): Option for the created dataset. ``train`` or ``test``.
    transform (callable, optional): A function/transform that takes in an PIL image
        and returns a transformed version.
    resize_size(int, int): Width and height of the images to be resized for the dataset.
    augment_data(bool): Flag to augment the data or not. If d_type is `test`, augmentation is
        disabled.
    """

    labels = ["human", "no_human"]
    label_to_id_map = {k: v for v, k in enumerate(labels)}
    label_to_folder_map = {"human": "with_human", "no_human": "without_human"}

    def __init__(
        self, root_dir, d_type, transform=None, resize_size=(224, 224), augment_data=False
    ):
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, "desert_15", d_type)
        if not self.__check_desert_data_exist():
            self.__print_download_manual()
            sys.exit("Dataset not found!")

        self.__get_image_paths()

        self.album_transform = None
        if d_type == "train" and augment_data:
            self.album_transform = album.Compose(
                [
                    album.GaussNoise(var_limit=(1.0, 20.0), p=0.25),
                    album.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                    album.ColorJitter(p=0.5),
                    album.SmallestMaxSize(max_size=int(1.2 * min(resize_size))),
                    album.ShiftScaleRotate(
                        shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                    ),
                    album.RandomCrop(height=resize_size[0], width=resize_size[1]),
                    album.HorizontalFlip(p=0.5),
                    album.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
                ]
            )
        if not augment_data or d_type == "test":
            self.album_transform = album.Compose(
                [
                    album.SmallestMaxSize(max_size=int(1.2 * min(resize_size))),
                    album.CenterCrop(height=resize_size[0], width=resize_size[1]),
                    album.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
                ]
            )

        self.transform = transform

    def __check_desert_data_exist(self):
        return os.path.isdir(self.data_dir)

    def __print_download_manual(self):
        print("******************************************")
        print("Malik says: make sure to set correct path")
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

        if self.album_transform:
            image = self.album_transform(image=image)["image"]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_desert_dataset(data, load_train, load_test):
    """
    Load the Cats vs Dogs dataset.
    Returns each datasample in 128x128 size.

    Data Augmentation: Train samples are augmented randomly with
        - Additive Gaussian Noise
        - RGB Shift
        - Color Jitter
        - Shift&Scale&Rotate
        - Random Crop
        - Horizontal Flip
    """
    (data_dir, args) = data

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            ai8x.normalize(args=args),
        ]
    )

    if load_train:
        train_dataset = Desert(
            root_dir=data_dir, d_type="train", transform=transform, augment_data=True
        )
    else:
        train_dataset = None

    if load_test:
        test_dataset = Desert(root_dir=data_dir, d_type="test", transform=transform)
    else:
        test_dataset = None

    return train_dataset, test_dataset


datasets = [
    {
        "name": "desert",
        "input": (3, 224, 224),
        "output": ("human", "no human"),
        "loader": get_desert_dataset,
    },
]
