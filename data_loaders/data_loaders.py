import os
from typing import Tuple, List, Optional

import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms.transforms import _setup_size

from constants import DEVICE, IMAGENET_VAL_DIR, IMAGENET_TRAIN_DIR
from image_processing_utils import create_valid_grid


def random_sampling(dset: Dataset, num_samples: int) -> Subset:
    """
    Selects a random subset of samples from the given dataset.

    Args:
        dset (Dataset): The dataset to sample from.
        num_samples (int): The number of samples to select.

    Returns:
        Subset: A subset of the original dataset containing the selected samples.
    """
    image_ids = np.random.choice(len(dset), num_samples, replace=False)
    return Subset(dset, image_ids)


def validation_preprocess() -> transforms.Compose:
    """
    Defines the preprocessing transformations for validation data.

    Returns:
        transforms.Compose: A composition of preprocessing transforms.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class RandomDataset(Dataset):
    """
    A PyTorch dataset that generates random Gaussian samples with mean 0 and variance 1.
    """

    def __init__(
        self,
        length: int,
        size: List[int],
        digitize: bool = True,
    ) -> None:
        """
        Initializes the RandomDataset.

        Args:
            length (int): The number of samples in the dataset.
            size (List[int]): The size of each sample (e.g., [3, 224, 224]).
            digitize (bool, optional): Whether to apply digitization to the samples. Defaults to True.
        """
        self.length = length
        self.size = size
        self.digitize = digitize
        self.valid_grid = create_valid_grid().cpu()

    def __len__(self) -> int:
        """
        Gets the length of the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Retrieves a sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            torch.Tensor: The generated random sample.
        """
        sample = torch.randn(self.size).cpu()
        if self.digitize:
            for i_ch in range(3):
                sample[i_ch, sample[i_ch, :, :] > self.valid_grid[i_ch, :].max()] = self.valid_grid[i_ch, :].max()
                sample[i_ch, sample[i_ch, :, :] < self.valid_grid[i_ch, :].min()] = self.valid_grid[i_ch, :].min()
        return sample.float()


class ImageFolderWithHiddenLabels(Dataset):
    """
    A PyTorch dataset that wraps around ImageFolder to hide labels unless explicitly requested.
    """

    def __init__(self, image_folder: datasets.ImageFolder) -> None:
        """
        Initializes the ImageFolderWithHiddenLabels.

        Args:
            image_folder (datasets.ImageFolder): The ImageFolder dataset to wrap.
        """
        self.image_folder = image_folder

    def __len__(self) -> int:
        """
        Gets the length of the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.image_folder)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Retrieves an image from the dataset without its label.

        Args:
            idx (int): The index of the image to retrieve.

        Returns:
            torch.Tensor: The image tensor.
        """
        return self.image_folder.__getitem__(idx)[0]

    def get_label(self, idx: int) -> int:
        """
        Retrieves the label of a specific image.

        Args:
            idx (int): The index of the image.

        Returns:
            int: The label of the image.
        """
        return self.image_folder.__getitem__(idx)[1]


def get_data_loader(
    dataset: Dataset,
    batch_size: int = 50,
    shuffle: bool = False,
    debug: bool = False
) -> DataLoader:
    """
    Creates a DataLoader for the given dataset.

    Args:
        dataset (Dataset): The dataset to load.
        batch_size (int, optional): The number of samples per batch. Defaults to 50.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
        debug (bool, optional): If True, uses fewer worker threads for debugging. Defaults to False.

    Returns:
        DataLoader: The DataLoader for the dataset.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0 if debug else 8,
        pin_memory=True
    )


def get_validation_loader(data_path: str, batch_size: int = 50, debug: bool = False) -> DataLoader:
    """
    Creates a DataLoader for the validation dataset.

    Args:
        data_path (str): The path to the validation data directory.
        batch_size (int, optional): The number of samples per batch. Defaults to 50.
        debug (bool, optional): If True, uses fewer worker threads for debugging. Defaults to False.

    Returns:
        DataLoader: The DataLoader for the validation dataset.
    """
    valdir = os.path.join(data_path, IMAGENET_VAL_DIR)
    preprocess = validation_preprocess()
    data_loader = DataLoader(
        datasets.ImageFolder(valdir, preprocess),
        batch_size=batch_size,
        shuffle=False,
        num_workers=16 if debug else 8,
        pin_memory=True
    )
    return data_loader


def get_real_data_dataloader(
    data_path: str,
    batch_size: int = 50,
    debug: bool = False,
    shuffle: bool = True,
    n_samples: Optional[int] = None
) -> torch.Tensor:
    """
    Creates a DataLoader for real data and optionally samples a subset of it.

    Args:
        data_path (str): The path to the training and validation data directories.
        batch_size (int, optional): The number of samples per batch. Defaults to 50.
        debug (bool, optional): If True, uses fewer worker threads for debugging. Defaults to False.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        n_samples (Optional[int], optional): The number of samples to select. If None, uses the entire dataset. Defaults to None.

    Returns:
        torch.Tensor: A tensor containing the concatenated images from the DataLoader.
    """
    traindir = os.path.join(data_path, IMAGENET_TRAIN_DIR)
    valdir = os.path.join(data_path, IMAGENET_VAL_DIR)
    images_folder = valdir if debug else traindir
    preprocess_fn = validation_preprocess()

    ds_images_folder = datasets.ImageFolder(images_folder, preprocess_fn)
    dataset = ImageFolderWithHiddenLabels(ds_images_folder)

    if n_samples is not None:
        dataset = random_sampling(dataset, num_samples=n_samples)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0 if debug else 8,
        pin_memory=True
    )

    images: Optional[torch.Tensor] = None
    for batch_imgs in data_loader:
        if images is None:
            images = batch_imgs
        else:
            images = torch.cat([images, batch_imgs])
    return images


def get_random_noise_dataset(
    n_images: int = 1000,
    size: Tuple[int, int] = (224, 224),
    batch_size: int = 50,
    digitize: bool = True
) -> DataLoader:
    """
    Creates a DataLoader for a dataset of random noise images.

    Args:
        n_images (int, optional): The number of random samples to generate. Defaults to 1000.
        size (Tuple[int, int], optional): The height and width of each sample. Defaults to (224, 224).
        batch_size (int, optional): The number of samples per batch. Defaults to 50.
        digitize (bool, optional): Whether to apply digitization to the samples. Defaults to True.

    Returns:
        DataLoader: The DataLoader for the random noise dataset.
    """
    image_size = list(_setup_size(size, error_msg="Please provide only two dimensions (h, w) for size."))
    dataset = RandomDataset(
        length=n_images,
        size=[3] + image_size,
        digitize=digitize
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True
    )
    return data_loader