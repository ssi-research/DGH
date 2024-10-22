from torch import Tensor
from torchvision.transforms import RandomCrop, RandomHorizontalFlip, CenterCrop
import torch
from typing import Any

from constants import DEVICE
from image_processing_utils import Smoothing, create_valid_grid


class BaseImagePipeline:
    def __init__(self, output_image_size: int, padding: int = 0) -> None:
        """
        Base class for image pipeline.

        Args:
            output_image_size (int): The desired output image size.
            padding (int, optional): Padding size for the image. Defaults to 0.
        """
        self.output_image_size = output_image_size
        self.padding = padding

    def get_image_input_size(self) -> int:
        """
        Get the size of the input image for the image pipeline.

        Returns:
            int: The input image size.
        """
        raise NotImplementedError

    def image_input_manipulation(self, images: Any) -> Any:
        """
        Perform image input manipulation in the image pipeline.

        Args:
            images (Any): Input images.

        Returns:
            Any: Manipulated images.
        """
        raise NotImplementedError

    def image_output_finalize(self, images: Any) -> Any:
        """
        Perform finalization of output images in the image pipeline.

        Args:
            images (Any): Output images.

        Returns:
            Any: Finalized images.
        """
        raise NotImplementedError

class AugmentationAndSmoothingImagePipeline(BaseImagePipeline):
    """
    An image pipeline implementation for PyTorch models.
    """

    def __init__(
        self,
        output_image_size: int,
        padding: int = 0,
        smoothing_filter_size: int = 3,
        smoothing_filter_sigma: float = 1.25,
        img_clipping: bool = True
    ) -> None:
        """
        Initialize the AugmentationAndSmoothingImagePipeline.

        Args:
            output_image_size (int): The desired output image size.
            padding (int, optional): Padding size for the image. Defaults to 0.
            smoothing_filter_size (int, optional): Size of the smoothing filter. Defaults to 3.
            smoothing_filter_sigma (float, optional): Sigma of the smoothing filter. Defaults to 1.25.
            img_clipping (bool, optional): Whether to apply image clipping. Defaults to True.
        """
        super(AugmentationAndSmoothingImagePipeline, self).__init__(output_image_size, padding)
        self.random_crop = RandomCrop(self.output_image_size)
        self.random_flip = RandomHorizontalFlip(0.5)
        self.center_crop = CenterCrop(self.output_image_size)
        self.smoothing = Smoothing(size=smoothing_filter_size, sigma=smoothing_filter_sigma)
        self.img_clipping = img_clipping
        self.valid_grid = create_valid_grid().to(DEVICE)

    def get_image_input_size(self) -> int:
        """
        Get the input size of the image.

        Returns:
            int: The input image size.
        """
        return self.output_image_size + self.padding

    def image_input_manipulation(self, images: Tensor) -> Tensor:
        """
        Manipulate the input images.

        Args:
            images (Tensor): The input images.

        Returns:
            Tensor: The manipulated images.
        """
        new_images = self.random_flip(images)
        new_images = self.smoothing(new_images)
        new_images  = self.random_crop(new_images)
        if self.img_clipping:
            new_images = self.clip_images(new_images, self.valid_grid)
        return new_images

    def image_output_finalize(self, images: Tensor) -> Tensor:
        """
        Finalize the output images.

        Args:
            images (Tensor): The output images.

        Returns:
            Tensor: The finalized images.
        """
        new_images = self.smoothing(images)
        new_images = self.center_crop(new_images)
        if self.img_clipping:
            new_images = self.clip_images(new_images, self.valid_grid)
        return new_images

    @staticmethod
    def clip_images(images: Tensor, valid_grid: Tensor, reflection: bool = False) -> Tensor:
        """
        Clips the images to lie within the valid grid, optionally applying reflection.

        Args:
            images (Tensor): The images to clip.
            valid_grid (Tensor): The valid grid values for clipping.
            reflection (bool, optional): Whether to apply reflection after clipping. Defaults to False.

        Returns:
            Tensor: The clipped images.
        """
        with torch.no_grad():
            for i_ch in range(valid_grid.shape[0]):
                clamp = torch.clamp(images[:, i_ch, :, :], valid_grid[i_ch, :].min(), valid_grid[i_ch, :].max())
                if reflection:
                    images[:, i_ch, :, :] = 2 * clamp - images[:, i_ch, :, :]
                else:
                    images[:, i_ch, :, :] = clamp
        return images
