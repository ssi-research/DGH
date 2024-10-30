# =============================================================================
# Source: https://github.com/yhhhli/BRECQ
# License: MIT License
#
# Attribution:
# This file was sourced from the repository "BRECQ" by Yuhang Li,
# available at https://github.com/yhhhli/BRECQ. Licensed under the MIT License.
# =============================================================================
import os
import pickle
import torch.nn as nn

from brecq.quant.block_recon import block_reconstruction
from brecq.quant.layer_recon import layer_reconstruction
from brecq.quant.quant_model import QuantModel, BaseQuantBlock
from constants import DEVICE
from brecq.quant.quant_layer import QuantModule

import torch

def run_model_on_images(model, images, initial_batch_size=1024):
    num_images = images.size(0)
    batch_size = initial_batch_size

    while batch_size > 0:
        try:
            batch = images[:batch_size]
            outputs = model(batch.to(DEVICE))
            print(f"Successfully processed with batch size: {batch_size}")
            break

        except RuntimeError as e:
            print(f"Out of memory with batch size: {batch_size}. Trying a smaller batch.")
            batch_size -= 64  # Reduce the batch size
            torch.cuda.empty_cache()  # Clear memory cache if using GPU

    else:
        raise RuntimeError("Unable to process images, even with the smallest batch size.")

def get_train_samples(train_loader, num_samples):
    train_data = []
    for batch in train_loader:
        train_data.append(batch)
        if len(train_data) * batch.size(0) >= num_samples:
            break
    return torch.cat(train_data, dim=0)[:num_samples]


def brecq(model, args, cali_data, evaluation_fn, val_loader):
    if args.activation_n_bits > 8:
        print(
            f'Changed activation bits from {args.activation_n_bits} to 8, since 8 is the max activation bits BREQC algorithm works with.')
        args.activation_n_bits = 8
    # train_loader, test_loader = build_imagenet_data(batch_size=args.batch_size, workers=args.workers, use_val_transform=args.use_val_transform)

    # build quantization parameters
    wq_params = {'n_bits': args.weights_n_bits, 'channel_wise': True, 'scale_method': 'mse'}
    aq_params = {'n_bits': args.activation_n_bits, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': args.act_quant}
    qnn = QuantModel(model=model, weight_quant_params=wq_params, act_quant_params=aq_params)
    qnn.to(DEVICE)
    qnn.eval()

    # if args.visualize_quant_fmodel:

    if args.head_stem_8bit:
        print('Setting the first and the last layer to 8-bit')
        qnn.set_first_last_layer_to_8bit()

    # Initialize weight quantization parameters
    qnn.set_quant_state(True, False)

    new_shape = (cali_data.size(0) * cali_data.size(1),) + cali_data.size()[2:]

    # Reshape the tensor
    reshaped_tensor = cali_data.view(new_shape)

    run_model_on_images(qnn, reshaped_tensor, 2 if args.debug else 512)

    # Kwargs for weight rounding calibration
    kwargs = dict(cali_data=cali_data, iters=4 if args.debug else args.num_iterations,
                  weight=args.weight, asym=True,
                  b_range=(args.b_start, args.b_end), warmup=args.warmup,
                  act_quant=False, opt_mode='mse',
                  batch_size=args.batch_size_for_calibration)

    def recon_model(model: nn.Module, name_prefix=''):
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for name, module in model.named_children():
            full_name = name_prefix + name
            if isinstance(module, QuantModule):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of layer {}'.format(full_name))
                    continue
                else:
                    print('Reconstruction for layer {}'.format(full_name))
                    layer_reconstruction(qnn, module, **kwargs)
            elif isinstance(module, BaseQuantBlock):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of block {}'.format(full_name))
                    continue
                else:
                    print('Reconstruction for block {}'.format(full_name))
                    if args.layer_reconstruction:
                        for child_name, child_module in module.named_modules():
                            recon_model(child_module, name_prefix=full_name + '_')
                    else:
                        block_reconstruction(qnn, module, **kwargs)
            else:
                recon_model(module, name_prefix=full_name + '_')

    # Start calibration
    recon_model(qnn)
    qnn.set_quant_state(weight_quant=True, act_quant=False)
    qnn.eval()
    if not args.debug:
        weight_only_quant_accuracy, _ = evaluation_fn(qnn, val_loader)
        print('Weight only quant accuracy, BRECQ: ', weight_only_quant_accuracy)
    else:
        print('Debug mode: not running weights only accuracy')


    if args.act_quant:
        # Initialize activation quantization parameters
        qnn.set_quant_state(True, True)
        with torch.no_grad():
            # _ = qnn(reshaped_tensor.to(DEVICE))
            if args.model_name in ['retinanet_resnet50_fpn', 'retinanet_resnet50_fpn_v2']:
                run_model_on_images(qnn, reshaped_tensor, 2)
            else:
                run_model_on_images(qnn, reshaped_tensor, 2 if args.debug else 512)
        # Disable output quantization because network output
        # does not get involved in further computation
        if args.disable_network_output_quantization:
            qnn.disable_network_output_quantization()
        # Kwargs for activation rounding calibration
        kwargs = dict(cali_data=cali_data, iters=5 if args.debug else args.iters_a,
                      act_quant=True, opt_mode='mse', lr=args.lr, p=args.p,
                      batch_size=args.batch_size_for_calibration)
        recon_model(qnn)
        qnn.set_quant_state(weight_quant=True, act_quant=True)
    return qnn, weight_only_quant_accuracy
