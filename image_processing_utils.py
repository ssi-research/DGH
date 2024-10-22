from typing import List

import torch.nn.functional as F
from constants import DEVICE
from torchvision.transforms.transforms import Normalize
import numpy as np
import torch
from constants import NORMALIZATION_IMAGENET_MEAN, NORMALIZATION_IMAGENET_STD

def create_valid_grid(means: List[float] = NORMALIZATION_IMAGENET_MEAN,
                      stds: List[float] = NORMALIZATION_IMAGENET_STD) -> torch.Tensor:
    """
    Create a valid grid for image normalization.

    Args:
        means (List[float]): Mean values for normalization.
        stds (List[float]): Standard deviation values for normalization.

    Returns:
        torch.Tensor: The valid grid for image normalization.
    """
    # Create a grid of pixel values from 0 to 255, normalized to [0, 1] range
    pixel_grid = torch.from_numpy(np.array(list(range(256))).repeat(3).reshape(-1, 3) / 255)

    # Normalize the pixel grid using the provided means and stds
    valid_grid = Normalize(mean=means, std=stds)(pixel_grid.transpose(1, 0)[None, :, :, None]).squeeze()

    # Move the valid grid to the specified device
    return valid_grid.to(DEVICE)


class Smoothing(torch.nn.Module):
    def __init__(self, size: int = 3, sigma: float = 1.0, kernel: torch.Tensor = None):
        """
        Initialize the Smoothing module.

        Args:
            size (int, optional): Size of the Gaussian kernel. Defaults to 3.
            sigma (float, optional): Standard deviation of the Gaussian kernel. Defaults to 1.0.
            kernel (torch.Tensor, optional): Predefined kernel. If None, a Gaussian kernel will be created.
        """
        super().__init__()
        if kernel is None:
            kernel = self.gaussian_kernel(size, sigma)

        # Reshape the kernel to be used with 2D convolution and repeat for 3 color channels
        kernel = kernel.view(1, 1, kernel.shape[0], kernel.shape[1]).repeat(3, 1, 1, 1)
        self.kernel = kernel

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply smoothing to the input image using the predefined kernel.

        Args:
            image (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Smoothed image tensor.
        """
        return F.conv2d(image, self.kernel.to(DEVICE), padding=self.kernel.shape[-1] // 2, groups=3)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(kernel={self.kernel.shape[-1]})"

    @staticmethod
    def gaussian_kernel(size: int = 3, sigma: float = 1.0) -> torch.Tensor:
        """
        Creates a Gaussian Kernel with the given size and sigma.

        Args:
            size (int, optional): Size of the Gaussian kernel. Defaults to 3.
            sigma (float, optional): Standard deviation of the Gaussian kernel. Defaults to 1.0.

        Returns:
            torch.Tensor: Gaussian kernel tensor.
        """
        # Create a coordinate grid
        axis = torch.arange(-size // 2 + 1., size // 2 + 1.)
        x, y = torch.meshgrid(axis, axis)

        # Calculate the Gaussian function
        kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

        # Normalize the kernel
        kernel = kernel / torch.sum(kernel)
        return kernel