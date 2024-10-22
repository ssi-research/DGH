import argparse
import time

from constants import DistillDataAlg
from data_loaders.data_loaders import get_validation_loader
from data_loaders.calibration_dataset import get_calibration_dataset
from constants import BatchNormAlignemntLossType, SchedularType
from model_params.model_dict import model_dict
from utils import display_run_time, set_seed
from brecq.brecq import brecq


def argument_handler():
    """
    Handles and parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='')

    ####################################
    # Running parameters
    ####################################
    parser.add_argument('--model_name', '-m', type=str, required=True,
                        help='Name of the model to run. Choose from the available models in the model dictionary.', choices=model_dict.keys())
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing data during calibration.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the ImageNet dataset directory.')
    parser.add_argument('--n_images', type=int, default=1, help='Number of images to use for calibration.')
    parser.add_argument('--random_seed', type=int, default=0, help='Random seed for initialization to ensure reproducibility.')
    parser.add_argument('--eval_float_accuracy', action='store_true', help='Evaluate the model using floating-point precision.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode for detailed logging and reduced performance.')

    ####################################
    # Quantization parameters
    ####################################
    parser.add_argument('--num_iterations', type=int, default=20000, help='Total number of iterations for the quantization algorithm.')
    parser.add_argument('--weights_n_bits', type=int, default=4, help='Number of bits to use for quantizing weights.')
    parser.add_argument('--activation_n_bits', type=int, default=4, help='Number of bits to use for quantizing activations.')
    parser.add_argument('--act_quant', action='store_true', help='Enable quantization of activations.')
    parser.add_argument('--head_stem_8bit', action='store_true', help='Quantize the first and last layers of the model to 8 bits (academic quantization).')
    parser.add_argument('--disable_network_output_quantization', action='store_true', help='Disable quantization of the model outputs (academic quantization).')
    parser.add_argument('--layer_reconstruction', action='store_true', help='Use only layer reconstruction for quantization, similar to AdaRound.')

    ####################################
    # Data Generation Parameters
    ####################################
    parser.add_argument('--calib_data_type', default=DistillDataAlg.DGH.value, type=str, choices=DistillDataAlg.get_values(),
                        help='Type of data to use for calibration. Options include Data Generation (DGH), Real Data, or Random Noise.')
    parser.add_argument('--distill_bn_loss_type', default=BatchNormAlignemntLossType.L2_SQUARE.value, type=str, choices=BatchNormAlignemntLossType.get_values(),
                        help='Type of BatchNorm alignment loss function to use.')
    parser.add_argument('--distill_scheduler_type', default=SchedularType.REDUCE_ON_PLATEAU_WITH_RESET.value, type=str, choices=SchedularType.get_values(),
                        help='Type of learning rate scheduler to use during image generation optimization.')
    parser.add_argument('--distill_image_padding', type=int, default=32, help='Padding size to apply to images during DGH preprocessing.')
    parser.add_argument('--distill_batch_size', type=int, default=128, help='Batch size for GPU processing during image generation (not related to optimization scope).')
    parser.add_argument('--distill_num_iter', default=1000, type=int, help='Number of iterations for the image generation process.')
    parser.add_argument('--output_loss_factor', default=3e-7, type=float, help='Factor to scale the output loss in the total loss calculation.')
    parser.add_argument('--distill_std_factor', default=10, type=float, help='Factor to scale the standard deviation component in the BatchNorm alignment loss.')
    parser.add_argument('--distill_smoothing_filter_size', default=3, type=int, help='Size of the smoothing filter used in DGH image preprocessing.')
    parser.add_argument('--distill_smoothing_filter_sigma', default=1.25, type=float, help='Sigma (standard deviation) of the smoothing filter used in DGH image preprocessing.')
    parser.add_argument('--distill_lr', default=16, type=float, help='Learning rate for optimizing image generation.')
    parser.add_argument('--img_size', default=224, type=int, help='Size of the images to generate.')

    ####################################
    # BRECQ Parameters
    ####################################
    # weight calibration parameters
    parser.add_argument('--batch_size_for_calibration', type=int, default=32, help='Batch size used during the calibration phase.')
    parser.add_argument('--weight', default=0.01, type=float, help='Weight balancing the rounding cost and the reconstruction loss in BRECQ.')
    parser.add_argument('--b_start', default=20, type=int, help='Initial temperature value at the start of calibration.')
    parser.add_argument('--b_end', default=2, type=int, help='Final temperature value at the end of calibration.')
    parser.add_argument('--warmup', default=0.2, type=float, help='Proportion of iterations to use as a warmup period where no regularization is applied.')
    parser.add_argument('--step', default=20, type=int, help='Number of steps after which to record SNN outputs.')

    # activation calibration parameters
    parser.add_argument('--iters_a', default=5000, type=int, help='Number of iterations for Learned Step Size Quantization (LSQ).')
    parser.add_argument('--lr', default=4e-4, type=float, help='Learning rate for LSQ optimization.')
    parser.add_argument('--p', default=2.4, type=float, help='Order of the L_p norm used for minimization in LSQ.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    #################################################
    # Main Initialization
    #################################################

    start_time = time.time()
    # Parse command-line arguments
    args = argument_handler()

    # Set random seed for reproducibility
    set_seed(args.random_seed)

    #################################################
    # Network and DataLoader Initialization
    #################################################

    # Retrieve model parameters based on the provided model name
    model_params = model_dict.get(args.model_name)
    # Initialize the model
    model = model_params.get_model()

    # Initialize the validation data loader
    val_loader = get_validation_loader(
        data_path=args.data_path,
        batch_size=model_params.get_batch_size(),
        debug=args.debug
    )

    #################################################
    # Run Float Model Accuracy Evaluation
    #################################################
    if args.eval_float_accuracy:
        t1 = time.time()
        # Evaluate the float (original) model's accuracy
        accuracy, _ = model_params.evaluation_fn(model, val_loader, args.debug)
        t2 = time.time()
        print(f'Float accuracy: {accuracy}%, evaluation time: {t2 - t1:.2f} seconds')
    else:
        # If not evaluating, print the pre-saved float accuracy
        print(f'Float accuracy from saved evaluation: {model_params.float_accuracy}%')

    #################################################
    # Generate Representative Dataset for Quantization
    #################################################
    start_time = time.time()
    # Generate calibration dataset using the DGH method
    generated_images = get_calibration_dataset(
        model=model,
        args=args,
        img_size=model_params.get_img_size(override_size=args.img_size)
    )
    print(f"Generated {generated_images.shape[0] * generated_images.shape[1]} images "
          f"using {args.distill_num_iter} iterations. Image generation time: {int(time.time() - start_time)} seconds")

    #################################################
    # Apply BRECQ Quantization
    #################################################
    # Perform BRECQ quantization on the model using the generated images
    quantized_model, weight_only_quant_accuracy = brecq(
        model=model,
        args=args,
        cali_data=generated_images,
        evaluation_fn=model_params.evaluation_fn,
        val_loader=val_loader
    )

    #################################################
    # Evaluate Quantized Model Accuracy
    #################################################
    # Evaluate the quantized model's accuracy
    quant_accuracy, quant_inds = model_params.evaluation_fn(quantized_model, val_loader)
    print('Top1 accuracy results on ImageNet1k validation set: ')
    print(f"----> Weight only quantization: {weight_only_quant_accuracy}%")
    print(f"----> Full quantization (weights + activations): {quant_accuracy}%")

    # Display the total runtime of the script
    display_run_time(start_time, time.time())

