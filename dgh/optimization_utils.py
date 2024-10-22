from typing import Callable, Type, Any, Dict, Tuple
import numpy as np
import torch
from torch import Tensor
from torch.cuda.amp import autocast
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from constants import IMAGE_INPUT, DEVICE
from image_processing_utils import create_valid_grid
from dgh.model_info_exctractors import ActivationExtractor, OrigBNStatsHolder


class AllImagesOptimizationHandler:
    """
    Handles the optimization process for all images, including image processing, optimizer scheduling,
    and batch statistics management.
    """
    def __init__(
        self,
        init_dataset: DataLoader,
        image_pipeline: Any,
        activation_extractor: ActivationExtractor,
        batch_size: int,
        optimizer: Callable,
        scheduler: Callable,
        initial_lr: float,
        eps: float = 1e-6
    ) -> None:
        """
        Initializes the AllImagesOptimizationHandler with the given parameters.

        Args:
            init_dataset (DataLoader): The initial dataset used for images generation.
            image_pipeline (Any): The image pipeline for processing images.
            activation_extractor (ActivationExtractor): Extractor for layer activations.
            batch_size (int): The size of each batch.
            optimizer (Callable): The optimizer constructor.
            scheduler (Callable): The scheduler constructor.
            initial_lr (float): The initial learning rate.
            eps (float, optional): A small value added for numerical stability. Defaults to 1e-6.
        """
        self.image_pipeline = image_pipeline
        self.batch_size = batch_size
        self.eps = eps
        self.mean_axis = [0, 2, 3]
        self.valid_grid = create_valid_grid()

        # Create BatchOptimizationHolder objects for each batch in the initial dataset
        self.batch_opt_holders_list = []
        self.imgs_ema_list = []
        self.images = None
        for data_input in init_dataset:
            if isinstance(data_input, list):
                batched_images, targets = data_input
            else:
                batched_images = data_input

            self.batch_opt_holders_list.append(
                BatchOptimizationHolder(
                    images=batched_images.to(DEVICE),
                    optimizer=optimizer,
                    scheduler=scheduler,
                    initial_lr=initial_lr))
            if self.images is None:
                self.images = batched_images
            else:
                self.images = torch.concatenate([self.images, batched_images], dim=0)
        self.optimizer = optimizer([self.images], lr=initial_lr)
        self.scheduler = scheduler(self.optimizer)
        self.n_batches = len(self.batch_opt_holders_list)
        self.random_batch_reorder()
        self.all_imgs_stats_holder = AllImgsStatsHolder(n_batches=self.n_batches,
                                                        batch_size=self.batch_size,
                                                        mean_axis=self.mean_axis)

        with autocast():
            for i_batch in range(self.n_batches):
                input_imgs = self.image_pipeline.image_output_finalize(self.get_images_by_batch_index(i_batch))
                output = activation_extractor.run_on_inputs(input_imgs)
                self.all_imgs_stats_holder.update_batch_stats(batch_index=i_batch,
                                                              input_imgs=input_imgs,
                                                              activation_extractor=activation_extractor,
                                                              to_differentiate=False)

    def random_batch_reorder(self) -> None:
        """
        Randomly reorders the batch indices.
        """
        self.rand_batch_inds = np.random.choice(self.n_batches, self.n_batches, replace=False)

    def get_random_batch_index(self, index: int) -> int:
        """
        Retrieves the random batch index at the specified position.

        Args:
            index (int): The position index.

        Returns:
            int: The randomly reordered batch index.
        """
        return self.rand_batch_inds[index]

    def get_images_by_batch_index(self, batch_index: int) -> Tensor:
        """
        Retrieves the images from the batch optimization holder at the given batch index.

        Args:
            batch_index (int): The index of the batch optimization holder.

        Returns:
            Tensor: The images in the specified batch.
        """
        return self.batch_opt_holders_list[batch_index].get_images()

    def get_optimizer_by_batch_index(self, batch_index: int) -> Optimizer:
        """
        Retrieves the optimizer for the specific batch specified by the batch index.

        Args:
            batch_index (int): The index of the batch.

        Returns:
            Optimizer: The optimizer associated with the specified batch.
        """
        return self.batch_opt_holders_list[batch_index].get_optimizer()

    def get_scheduler_by_batch_index(self, batch_index: int)-> Any:
        """
        Retrieves the scheduler for the specific batch specified by the batch index.

        Args:
            batch_index (int): The index of the batch.

        Returns:
            Any: The scheduler associated with the specified batch.
        """
        return self.batch_opt_holders_list[batch_index].get_scheduler()

    def get_accumulated_stats_per_layer(self, layer_name: str) -> Tuple[Tensor, Tensor]:
        """
        Retrieves the accumulated activation statistics for a specific layer.

        Args:
            layer_name (str): The name of the layer.

        Returns:
            Tuple[Tensor, Tensor]: The averaged mean and variance of activations for the specified layer.
        """
        total_mean, total_second_moment = 0, 0
        for i_batch in range(self.n_batches):
            mean, second_moment, _ = self.all_imgs_stats_holder.get_stats(i_batch, layer_name)
            total_mean += mean
            total_second_moment += second_moment

        total_mean /= self.n_batches
        total_second_moment /= self.n_batches
        total_var = total_second_moment - torch.pow(total_mean, 2)
        return total_mean, total_var

    def compute_bn_loss(self,
                        input_imgs: Tensor,
                        batch_index: int,
                        activation_extractor: ActivationExtractor,
                        orig_bn_stats_holder: OrigBNStatsHolder,
                        bn_loss_fn: Callable,
                        layer_weights: Dict,
                        std_factor: float=1) -> Tensor:
        """
        Computes the batch normalization alignment loss.

        Args:
            input_imgs (Tensor): The input images.
            batch_index (int): The index of the batch.
            activation_extractor (ActivationExtractor): Extractor for layer activations.
            orig_bn_stats_holder (OrigBNStatsHolder): Holder for original BatchNorm statistics.
            bn_loss_fn (Callable): The batch norm alignment loss function.
            layer_weights (Dict[str, float]): Weights for each layer's loss contribution.
            std_factor (float, optional): Factor to scale the variance loss. Defaults to 1.0.

        Returns:
            Tensor: The computed batch norm alignment loss.
        """
        # Update the batch statistics for the current batch
        self.all_imgs_stats_holder.update_batch_stats(batch_index=batch_index,
                                                      input_imgs=input_imgs,
                                                      activation_extractor=activation_extractor,
                                                      to_differentiate=True)

        # Initialize variables for accumulating mean and variance differences
        total_mean_diff, total_var_diff = 0, 0
        mean_diff_per_layer, std_diff_per_layer = [], [] # for debug

        # Iterate over each BN layer
        for layer_name in orig_bn_stats_holder.get_bn_layer_names():
            # Get the layer weight for the current BN layer
            layer_weight = layer_weights.get(layer_name)
            if layer_weight is None:
                layer_weight = 0

            # Get the mean and variance from the original BN statistics
            bn_layer_mean = orig_bn_stats_holder.get_mean(layer_name)
            bn_layer_var = orig_bn_stats_holder.get_var(layer_name)

            imgs_layer_mean, imgs_layer_var = self.get_accumulated_stats_per_layer(layer_name)

            # Calculate the standard deviation from the variance
            eps = self.eps
            bn_layer_std = torch.sqrt(bn_layer_var + eps)
            while torch.any(torch.isnan(bn_layer_std)):
                eps *=10
                bn_layer_std = torch.sqrt(bn_layer_var + eps)
            eps = self.eps
            imgs_layer_std = torch.sqrt(imgs_layer_var + eps)
            while torch.any(torch.isnan(imgs_layer_std)):
                eps *= 10
                imgs_layer_std = torch.sqrt(imgs_layer_var + eps)

            # Accumulate the mean and variance loss metrics weighted by the layer weight
            mean_diff = bn_loss_fn(bn_layer_mean, imgs_layer_mean)
            std_diff = bn_loss_fn(bn_layer_std, imgs_layer_std)
            total_mean_diff += layer_weight * mean_diff
            total_var_diff += layer_weight * std_diff

            # mean_diff_per_layer.append(torch.linalg.norm(bn_layer_mean - imgs_layer_mean) ** 2 / torch.linalg.norm(bn_layer_mean) ** 2 )
            # std_diff_per_layer.append(torch.linalg.norm(bn_layer_std - imgs_layer_std) ** 2 / torch.linalg.norm(bn_layer_std) ** 2 )
            mean_diff_per_layer.append(mean_diff)
            std_diff_per_layer.append(std_diff)
        # Compute the total BN loss as the sum of mean and variance differences
        total_bn_loss = total_mean_diff + std_factor * total_var_diff

        return total_bn_loss.to(DEVICE)

    def update_statistics(
            self,
            input_imgs: Tensor,
            batch_index: int,
            activation_extractor: ActivationExtractor) -> None:
        """
        Updates the statistics for the images at the specified batch index.

        Args:
            input_imgs (Tensor): The input images.
            batch_index (int): The index of the batch.
            activation_extractor (ActivationExtractor): Extractor for layer activations.
        """
        self.all_imgs_stats_holder.update_batch_stats(batch_index=batch_index,
                                                      input_imgs=input_imgs,
                                                      activation_extractor=activation_extractor,
                                                      to_differentiate=False)

    def clip_images(self, batch_index: int, reflection: bool = False) -> None:
        """
        Clips the images in the specified batch to lie within the valid grid.

        Args:
            batch_index (int): The index of the batch.
            reflection (bool, optional): Whether to apply reflection after clipping. Defaults to False.
        """
        self.batch_opt_holders_list[batch_index].clip_images(self.valid_grid, reflection)

    def get_finalized_images(self) -> torch.Tensor:
        """
        Finalizes and retrieves all optimized images.

        Returns:
            Tensor: A tensor containing all finalized optimized images.
        """
        finalized_images = []
        finalized_images = None

        # Iterate over each batch
        for i_batch in range(self.n_batches):
            # Retrieve the images for the current batch
            batch_imgs = self.get_images_by_batch_index(i_batch)

            # Apply the image_pipeline's image_output_finalize method to finalize the batch of images
            finalized_batch = self.image_pipeline.image_output_finalize(batch_imgs).detach().clone().cpu()

            # Split the finalized batch into individual images and add them to the finalized_images list
            if finalized_images is None:
                finalized_images = finalized_batch
            else:
                finalized_images = torch.cat([finalized_images, finalized_batch])

        return finalized_images


class BatchOptimizationHolder:
    """
    Holds optimization-related information for a single batch of images.
    """
    def __init__(self,
                 images: Tensor,
                 optimizer: Optimizer,
                 scheduler: Any,
                 initial_lr: float)-> None:
        """
        Initializes the BatchOptimizationHolder with the given images, optimizer, and scheduler.

        Args:
            images (Tensor): A tensor containing the input images.
            optimizer (Callable): The optimizer constructor.
            scheduler (Callable): The scheduler constructor.
            initial_lr (float): The initial learning rate for the optimizer.
        """
        self.images = images
        self.images.requires_grad = True
        self.optimizer = optimizer([self.images], lr=initial_lr)
        self.scheduler = scheduler(self.optimizer)

    def clip_images(self, valid_grid: Tensor, reflection: bool = False) -> None:
        """
        Clips the images to lie within the valid grid, optionally applying reflection.

        Args:
            valid_grid (Tensor): The valid range grid for each channel.
            reflection (bool, optional): Whether to apply reflection after clipping. Defaults to False.
        """
        with torch.no_grad():
            for i_ch in range(valid_grid.shape[0]):
                clamp = torch.clamp(self.images[:, i_ch, :, :], valid_grid[i_ch, :].min(), valid_grid[i_ch, :].max())
                if reflection:
                    self.images[:, i_ch, :, :] = 2 * clamp - self.images[:, i_ch, :, :]
                else:
                    self.images[:, i_ch, :, :] = clamp
        self.images.requires_grad = True

    def get_images(self) -> Tensor:
        """
        Retrieves the stored images.

        Returns:
            Tensor: The images' tensor.
        """
        return self.images

    def get_optimizer(self) -> Optimizer:
        """
        Retrieves the optimizer associated with the batch.

        Returns:
            Optimizer: The optimizer instance.
        """
        return self.optimizer

    def get_scheduler(self) -> Any:
        """
        Retrieves the scheduler associated with the batch.

        Returns:
            Any: The scheduler instance.
        """
        return self.scheduler


class AllImgsStatsHolder:
    """
   Manages and accumulates statistics for all image batches across different layers.
   """
    def __init__(self,
                 n_batches: int,
                 batch_size: int,
                 mean_axis=Type[list]) -> None:
        """
        Initializes the AllImgsStatsHolder with the given parameters.

        Args:
            n_batches (int): The number of batches.
            batch_size (int): The size of each batch.
            mean_axis (List[int]): The axes along which to compute the mean.
            eps (float, optional): A small value added for numerical stability. Defaults to 1e-6.
        """
        self.mean_axis = mean_axis
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.batches_stats_holder_list = [BatchStatsHolder(self.mean_axis) for _ in range(self.n_batches)]
        self.data_bn_mean_all_batches_except_one = {}
        self.data_bn_second_moment_all_batches_except_one = {}
        self.bn_mean_all_batches = {}
        self.bn_second_moment_all_batches = {}

    def update_batch_stats(self,
                           batch_index: int,
                           input_imgs: Tensor,
                           activation_extractor: ActivationExtractor,
                           to_differentiate=False)-> None:
        """
        Updates the batch statistics for a given batch.

        Args:
            batch_index (int): The index of the batch.
            input_imgs (Tensor): The input images for which to calculate the statistics.
            activation_extractor (ActivationExtractor): The activation extractor object.
            to_differentiate (bool): Whether to enable gradient computation. Defaults to False.
        """
        self.batches_stats_holder_list[batch_index].clear()
        self.batches_stats_holder_list[batch_index].calc_bn_stats_from_activations(input_imgs=input_imgs,
                                                                                   activation_extractor=activation_extractor,
                                                                                   to_differentiate=to_differentiate)

    def get_stats(self,
                  batch_index: int,
                  layer_name: str) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Retrieves the statistics for a given batch and layer.

        Args:
            batch_index (int): The index of the batch.
            layer_name (str): The name of the layer.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: The mean, second moment, and variance for the specified batch and layer.
        """
        mean = self.batches_stats_holder_list[batch_index].get_mean(layer_name)
        second_moment = self.batches_stats_holder_list[batch_index].get_second_moment(layer_name)
        var = self.batches_stats_holder_list[batch_index].get_var(layer_name)
        return mean, second_moment, var


class BatchStatsHolder(object):
    """
    Holds statistical data for a single batch, including means and second moments for each layer.
    """
    def __init__(self,
                 mean_axis: Type[list],
                 eps: float = 1e-6)-> None:
        """
        Initializes the BatchStatsHolder with the given parameters.

        Args:
            mean_axis (List[int]): The axes along which to compute the mean.
            eps (float, optional): A small value added to avoid division by zero. Defaults to 1e-6.
        """
        self.eps = eps
        self.mean_axis = mean_axis
        self.bn_mean = {}
        self.bn_second_moment = {}

    def get_mean(self, bn_layer_name: str) -> Tensor:
        """
        Retrieves the mean for the specified BatchNorm layer.

        Args:
            bn_layer_name (str): The name of the BatchNorm layer.

        Returns:
            Tensor: The mean tensor for the specified layer.
        """
        return self.bn_mean[bn_layer_name]

    def get_second_moment(self, bn_layer_name: str) -> Tensor:
        """
        Retrieves the second moment for the specified BatchNorm layer.

        Args:
            bn_layer_name (str): The name of the BatchNorm layer.

        Returns:
            Tensor: The second moment tensor for the specified layer.
        """
        return self.bn_second_moment[bn_layer_name]

    def get_var(self, bn_layer_name: str) -> Tensor:
        """
        Calculates the variance for the specified BatchNorm layer.

        Args:
            bn_layer_name (str): The name of the BatchNorm layer.

        Returns:
            Tensor: The variance tensor for the specified layer.
        """
        mean = self.get_mean(bn_layer_name)
        second_moment = self.get_second_moment(bn_layer_name)
        var = second_moment - torch.pow(mean, 2.0)
        return var

    def update_layer_stats(self,
                           bn_layer_name: str,
                           mean: Tensor,
                           second_moment: Tensor)-> None:
        """
        Updates the statistics for a specific BatchNorm layer.

        Args:
            bn_layer_name (str): The name of the BatchNorm layer.
            mean (Tensor): The mean value for the layer.
            second_moment (Tensor): The second moment value for the layer.
        """
        self.bn_mean.update({bn_layer_name: mean})
        self.bn_second_moment.update({bn_layer_name: second_moment})

    def calc_bn_stats_from_activations(self,
                                       input_imgs: Tensor,
                                       activation_extractor: ActivationExtractor,
                                       to_differentiate: bool)-> None:
        """
        Calculates BatchNorm statistics from activations derived from input images.

        Args:
            input_imgs (Tensor): The input images tensor for which to calculate the statistics.
            activation_extractor (ActivationExtractor): The activation extractor object.
            to_differentiate (bool): Flag indicating whether to track gradients.
        """
        imgs_mean = torch.mean(input_imgs, dim=self.mean_axis)
        imgs_second_moment = torch.mean(torch.pow(input_imgs, 2.0), dim=self.mean_axis)
        if not to_differentiate:
            imgs_mean = imgs_mean.detach()
            imgs_second_moment = imgs_second_moment.detach()
        self.update_layer_stats(IMAGE_INPUT, imgs_mean, imgs_second_moment)
        # Extract statistics of intermediate convolution outputs before the BatchNorm layers
        for bn_layer_name in activation_extractor.get_bn_layer_names():
            bn_input_activations = activation_extractor.get_activation(bn_layer_name)
            if not to_differentiate:
                bn_input_activations = bn_input_activations.detach()

            collected_mean = torch.mean(bn_input_activations, dim=self.mean_axis)
            collected_second_moment = torch.mean(torch.pow(bn_input_activations, 2.0), dim=self.mean_axis)
            # prevent overflow due to float16. Max value for float16 is 65504
            collected_second_moment[collected_second_moment.isinf()] = 65504
            self.update_layer_stats(bn_layer_name, collected_mean, collected_second_moment)

    def clear(self)-> None:
        """
        Clears all stored statistics.
        """
        self.bn_mean.clear()
        self.bn_second_moment.clear()
        self.bn_mean = {}
        self.bn_second_moment = {}
        torch.cuda.empty_cache()