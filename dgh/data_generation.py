from typing import Tuple, Callable, Optional

import torch
from torch import Tensor, nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import RAdam, Optimizer
from constants import BATCHNORM_PYTORCH_LAYERS, DEVICE
from data_loaders.data_loaders import get_random_noise_dataset
from dgh.output_loss import compute_output_loss
from dgh.default_functions import get_bn_loss_fn, get_scheduler_step_fn
from dgh.image_pipeline import AugmentationAndSmoothingImagePipeline, BaseImagePipeline
from dgh.layer_weighting_functions import get_layer_weighting_function
from dgh.model_info_exctractors import ActivationExtractor, OrigBNStatsHolder
from dgh.optimization_utils import AllImagesOptimizationHandler
from utils import set_model

def data_generation(
    model: nn.Module,
    layer_weighting_type: Optional[str] = None,
    bn_loss_type: Optional[str] = None,
    scheduler_type: Optional[str] = None,
    optimizer: Callable[..., Optimizer] = RAdam,
    initial_lr: float = 16.0,
    n_iter: int = 1000,
    n_images: int = 1024,
    output_image_size: int = 224,
    image_pipeline_type: Callable[..., BaseImagePipeline] = AugmentationAndSmoothingImagePipeline,
    batch_size: int = 64,
    image_padding: int = 32,
    bn_layer_types: Tuple[type, ...] = BATCHNORM_PYTORCH_LAYERS,
    output_loss_factor: float = 1e-7,
    std_factor: float = 10.0,
    smoothing_filter_size: int = 3,
    smoothing_filter_sigma: float = 1.25,
) -> Tensor:
    """
    Generates data for hardware-friendly post-training quantization.

    This function generates calibration data by optimizing images to align with the original BatchNorm
    statistics of the provided model.

    Args:
        model (nn.Module): The model to be used for data generation.
        layer_weighting_type (Optional[str], optional): The type of layer weighting function. Defaults to None.
        bn_loss_type (Optional[str], optional): The type of batch normalization loss function. Defaults to None.
        scheduler_type (Optional[str], optional): The type of scheduler to use. Defaults to None.
        optimizer (Callable[..., Optimizer], optional): The optimizer class. Defaults to RAdam.
        initial_lr (float, optional): The initial learning rate. Defaults to 16.0.
        n_iter (int, optional): Number of iterations for data generation. Defaults to 1000.
        n_images (int, optional): Number of images to generate. Defaults to 1024.
        output_image_size (int, optional): Size of the output images. Defaults to 224.
        image_pipeline_type (Callable[..., BaseImagePipeline], optional): The image pipeline class. Defaults to AugmentationAndSmoothingImagePipeline.
        batch_size (int, optional): Batch size for data generation. Defaults to 64.
        image_padding (int, optional): Padding to apply to images. Defaults to 32.
        bn_layer_types (Tuple[type, ...], optional): List of batch normalization layer types. Defaults to BATCHNORM_PYTORCH_LAYERS.
        output_loss_factor (float, optional): Factor for output loss calculation. Defaults to 1e-7.
        std_factor (float, optional): Standard deviation factor for loss calculation. Defaults to 10.0.
        smoothing_filter_size (int, optional): Size of the smoothing filter. Defaults to 3.
        smoothing_filter_sigma (float, optional): Sigma value for the smoothing filter. Defaults to 1.25.

    Returns:
        Tensor: The generated images.
    """

    # Initialize functions and handlers
    layer_weighting_fn = get_layer_weighting_function(layer_weighting_type)
    scheduler_step_fn, scheduler = get_scheduler_step_fn(scheduler_type, n_iter)
    bna_loss_fn = get_bn_loss_fn(bn_loss_type)
    generated_images = None

    # Initialize mixed-precision scaler
    scaler = GradScaler()

    # Set the current model
    set_model(model)

    # Create an image pipeline object using the specified output_image_size and image_padding
    image_pipeline = image_pipeline_type(
        output_image_size,
        image_padding,
        smoothing_filter_size,
        smoothing_filter_sigma
    )

    # Create an activation extractor object to extract activations from the model
    activation_extractor = ActivationExtractor(model, bn_layer_types)

    # Create an orig_bn_stats_holder object to hold original BatchNorm statistics
    orig_bn_stats_holder = OrigBNStatsHolder(model, bn_layer_types)

    # Initialize the dataset for data generation
    init_dataset = get_random_noise_dataset(
        n_images=n_images,
        size=image_pipeline.get_image_input_size(),
        batch_size=batch_size)

    # Create an AllImagesOptimizationHandler object for handling optimization
    all_imgs_opt_handler = AllImagesOptimizationHandler(
        init_dataset=init_dataset,
        image_pipeline=image_pipeline,
        activation_extractor=activation_extractor,
        batch_size=batch_size,
        optimizer=optimizer,
        scheduler=scheduler,
        initial_lr=initial_lr,
    )

    # Define the log intervals for iterations
    iter_log_interval = list(range(0, n_iter, max(1, int(n_iter / 100))))

    # Perform data generation iterations
    for i_iter in range(n_iter):

        # Randomly reorder the batches
        all_imgs_opt_handler.random_batch_reorder()

        # Iterate over each batch
        for i_batch in range(all_imgs_opt_handler.n_batches):
            # Get the random batch index
            random_batch_index = all_imgs_opt_handler.get_random_batch_index(i_batch)

            # Get the images to optimize and the optimizer for the batch
            imgs_to_optimize = all_imgs_opt_handler.get_images_by_batch_index(random_batch_index)
            optimizer = all_imgs_opt_handler.get_optimizer_by_batch_index(random_batch_index)
            scheduler = all_imgs_opt_handler.get_scheduler_by_batch_index(random_batch_index)

            # Zero gradients
            optimizer.zero_grad()
            model.zero_grad()

            # Perform image input manipulation
            input_imgs = image_pipeline.image_input_manipulation(imgs_to_optimize)

            with autocast():
                output = activation_extractor.run_on_inputs(input_imgs)

            # Compute the layer weights based on orig_bn_stats_holder
            layer_weights = layer_weighting_fn(orig_bn_stats_holder)

            # Compute the total BatchNorm loss
            bn_loss_total = all_imgs_opt_handler.compute_bn_loss(input_imgs=input_imgs,
                                                                 batch_index=random_batch_index,
                                                                 activation_extractor=activation_extractor,
                                                                 orig_bn_stats_holder=orig_bn_stats_holder,
                                                                 bn_loss_fn=bna_loss_fn,
                                                                 layer_weights=layer_weights,
                                                                 std_factor=std_factor)
            # Compute the output loss
            if output_loss_factor > 0:
                output_loss = compute_output_loss(
                    orig_bn_stats_holder=orig_bn_stats_holder,
                    activation_extractor=activation_extractor)
            else:
                output_loss = torch.zeros(1).to(DEVICE)

            # Calculate the total loss
            total_loss = bn_loss_total + output_loss_factor * output_loss

            # Backpropagation and optimization
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Perform scheduler step
            scheduler_step_fn(scheduler, i_iter, total_loss.item())

            # Update the statistics based on the updated images
            with autocast():
                final_imgs = image_pipeline.image_output_finalize(imgs_to_optimize)
                all_imgs_opt_handler.update_statistics(input_imgs=final_imgs,
                                                       batch_index=random_batch_index,
                                                       activation_extractor=activation_extractor)

            # Log iteration progress
            if i_iter in iter_log_interval and i_batch == (all_imgs_opt_handler.n_batches - 1):
                print(f"Iteration {i_iter}/{n_iter}: "
                      f"Total Loss: {total_loss.item():.5f}, "
                      f"BN Loss: {bn_loss_total.item():.5f}, "
                      f"Output Loss: {output_loss.item():.5f}")

    # Retrieve the finalized images
    if generated_images is None:
        generated_images = all_imgs_opt_handler.get_finalized_images()
    else:
        generated_images = torch.cat([generated_images, all_imgs_opt_handler.get_finalized_images()])
    return generated_images
