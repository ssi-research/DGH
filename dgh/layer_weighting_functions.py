from typing import Optional, Callable, Dict, Any

from constants import LayerWeightingType
from dgh.model_info_exctractors import OrigBNStatsHolder


def average_layer_weighting(orig_bn_stats_holder: OrigBNStatsHolder, **kwargs: Any) -> Dict[str, float]:
    """
    Calculates average weighting for each batch normalization layer.

    Args:
        orig_bn_stats_holder (OrigBNStatsHolder): Holder containing original BN statistics.
        **kwargs (Any): Additional keyword arguments.

    Returns:
        Dict[str, float]: A dictionary mapping BN layer names to their average weights.
    """
    num_bn_layers = orig_bn_stats_holder.get_num_bn_layers()
    return {bn_layer_name: 1 / num_bn_layers for bn_layer_name in orig_bn_stats_holder.get_bn_layer_names()}


def get_layer_weighting_function(layer_weighting_type: Optional[str]) -> Callable[..., Dict[str, float]]:
    """
    Retrieves the layer weighting function based on the specified weighting type.

    Args:
        layer_weighting_type (Optional[str]): The type of layer weighting to retrieve.

    Returns:
        Callable[..., Dict[str, float]]: The layer weighting function.

    Raises:
        Exception: If an invalid layer weighting type is specified.
    """
    if layer_weighting_type is None or layer_weighting_type == LayerWeightingType.AVERAGE.value:
        return average_layer_weighting
    else:
        raise Exception(f"Unknown 'layer_weighting_type' {layer_weighting_type}. Please choose one of {LayerWeightingType.get_values()}")