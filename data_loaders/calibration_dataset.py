from typing import Any, Tuple

import torch

from constants import DistillDataAlg
from data_loaders.data_loaders import get_real_data_dataloader, get_random_noise_dataset
from dgh.data_generation import data_generation


def get_calibration_dataset(
        model: torch.nn.Module,
        args: Any,
        img_size: Tuple[int, int]
) -> torch.Tensor:
    """
    Generates a calibration dataset based on the specified calibration data type.

    Depending on the calibration data type specified in `args`, this function will generate
    synthetic data using data generation algorithms, use real data from a data loader,
    or generate random noise data.

    Args:
        model (torch.nn.Module): The PyTorch model used for data generation.
        args (Any): An object containing the necessary arguments for data generation,
                    such as calibration data type, paths, and parameters.
        img_size (Tuple[int, int]): The size of the images to generate or load.

    Returns:
        torch.Tensor: A tensor containing the generated or loaded calibration images,
                      reshaped for calibration purposes.

    Raises:
        Exception: If an invalid calibration dataset type is specified.
    """

    if args.calib_data_type == DistillDataAlg.DGH.value:
        generated_images = data_generation(
            model=model,
            bn_loss_type=args.distill_bn_loss_type,
            scheduler_type=args.distill_scheduler_type,
            initial_lr=args.distill_lr,
            n_iter=args.distill_num_iter,
            n_images=args.n_images,
            output_image_size=img_size,
            batch_size=args.distill_batch_size,
            image_padding=args.distill_image_padding,
            output_loss_factor=args.output_loss_factor,
            std_factor=args.distill_std_factor,
            smoothing_filter_size=args.distill_smoothing_filter_size,
            smoothing_filter_sigma=args.distill_smoothing_filter_sigma,
            )
    elif args.calib_data_type == DistillDataAlg.REAL_DATA.value:
        print("Using real data")

        generated_images = get_real_data_dataloader(
            data_path=args.data_path,
            batch_size=args.batch_size,
            debug=args.debug,
            shuffle=True,
            n_samples=args.n_images)

    elif args.calib_data_type == DistillDataAlg.RANDOM_NOISE.value:
        generated_images = get_random_noise_dataset(
            batch_size=args.generate_data_batch_size,
            n_images=args.n_images,
            size=img_size)
    else:
        raise Exception('Invalid dataset type')
    batch_size_for_calibration = min(args.n_images, args.batch_size_for_calibration)
    return generated_images.view(*(-1, batch_size_for_calibration, *generated_images.shape[1:]))