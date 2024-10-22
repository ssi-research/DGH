import torch
from constants import DEVICE
from dgh.model_info_exctractors import ActivationExtractor, OrigBNStatsHolder


def compute_output_loss(
        orig_bn_stats_holder: OrigBNStatsHolder,
        activation_extractor: ActivationExtractor) -> torch.Tensor:
    """
    Compute the output loss based on the activations and BatchNorm statistics.

    Args:
        orig_bn_stats_holder (OrigBNStatsHolder): Object holding original BatchNorm statistics.
        activation_extractor (ActivationExtractor): Object extracting activations from the model.

    Returns:
        torch.Tensor: The computed output loss.
    """
    # Extract the input and output activations of the output layers
    output_layers_inputs = activation_extractor.get_output_layer_input_activation()
    output_layers_output = activation_extractor.get_output_layer_output_activation()

    # Initialize the output loss to zero
    output_loss = torch.zeros(1).to(DEVICE)

    # Retrieve the last BatchNorm layer statistics
    last_bn_layer_name = orig_bn_stats_holder.get_bn_layer_names()[-1]
    last_bn_layer_mean = orig_bn_stats_holder.get_mean(last_bn_layer_name)
    last_bn_layer_std = orig_bn_stats_holder.get_std(last_bn_layer_name)

    # Extract the activation of the last BatchNorm layer
    last_bn_layer_activation = activation_extractor.get_activation(last_bn_layer_name)

    # Calculate the mean and standard deviation of the last BatchNorm layer activation
    last_bn_activation_mean = torch.mean(last_bn_layer_activation, dim=[2, 3])
    last_bn_activation_std = torch.std(last_bn_layer_activation, dim=[2, 3])

    # Iterate over the input and output activations of the output layers
    for last_layer_input, last_layers_output in zip(output_layers_inputs, output_layers_output):
        # Reshape the output to a 2D tensor
        output = torch.reshape(last_layers_output, [last_layers_output.shape[0], -1])

        # Calculate the maximum and minimum values of the output
        out_max, out_argmax = torch.max(output, dim=1)
        out_min, out_argmin = torch.min(output, dim=1)

        # Calculate the constraint mean and standard deviation
        constraint_mean = torch.maximum(
            torch.linalg.norm(last_bn_activation_mean - last_bn_layer_mean, dim=1) ** 2 / last_bn_layer_mean.shape[
                0] - 0.5, torch.zeros(last_bn_activation_mean.shape[0]).to(DEVICE))
        constraint_std = torch.maximum(
            torch.linalg.norm(last_bn_activation_std - last_bn_layer_std, dim=1) ** 2 / last_bn_layer_std.shape[
                0] - 0.5, torch.zeros(last_bn_activation_std.shape[0]).to(DEVICE))

        # Calculate the difference between the maximum and minimum output values
        out_max_min = out_max - out_min

        # Update the output loss
        output_loss += torch.mean(-out_max_min + constraint_mean + constraint_std)

    # Return the computed output loss
    return output_loss.to(DEVICE)
