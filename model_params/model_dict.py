import torch

from brecq import hubconf
from model_params.model_configs import PytorchModelParams

model_dict = {
    'resnet18': PytorchModelParams(
        model=hubconf.resnet18(pretrained=True),
        float_accuracy=71.08,
        img_size=224,
        max_batch_for_data_gen=256),
    'resnet50': PytorchModelParams(
        model=hubconf.resnet50(pretrained=True),
        float_accuracy=76.626,
        img_size=224,
        max_batch_for_data_gen=64),
    'mobilenet_v2': PytorchModelParams(
        model=hubconf.mobilenetv2(pretrained=True),
        float_accuracy=72.632,
        img_size=224,
        max_batch_for_data_gen=128),
}


def get_model(model_name: str) -> torch.nn.Module:
    """
    Retrieves a PyTorch model by its name.

    Args:
        model_name (str): The name of the model to retrieve.

    Returns:
        torch.nn.Module: The PyTorch model corresponding to the given name.
    """
    model_params = model_dict.get(model_name)
    model = model_params.get_model()
    return model
