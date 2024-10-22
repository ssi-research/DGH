import os
import random
import numpy as np
import torch
from contextlib import contextmanager, ContextDecorator
import time


def display_run_time(start_time: float, end_time: float) -> None:
    """
    Displays the total runtime in HH:MM:SS format.
    """
    total_run_time_in_sec = int(end_time - start_time)
    hours = int(total_run_time_in_sec / 3600)
    minutes = int(total_run_time_in_sec / 60 - hours * 60)
    seconds = int(total_run_time_in_sec - hours * 3600 - minutes * 60)
    if hours == 0:
        hours_disp = '00'
    elif hours < 10:
        hours_disp = '0' + str(hours)
    else:
        hours_disp = str(hours)
    if minutes == 0:
        minutes_disp = '00'
    elif minutes < 10:
        minutes_disp = '0' + str(minutes)
    else:
        minutes_disp = str(minutes)
    if seconds == 0:
        seconds_disp = '00'
    elif seconds < 10:
        seconds_disp = '0' + str(seconds)
    else:
        seconds_disp = str(seconds)
    hours_disp = '0' + hours_disp if len(hours_disp) < 2 else hours_disp
    print(f'Total run time in HH:MM:SS is: {hours_disp}:{minutes_disp}:{seconds_disp}')


def set_seed(seed: int = 0) -> None:
    """
    Sets the seed for various random number generators to ensure reproducibility.
    """
    print("Setting initial seed... ")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a PyTorch tensor to a NumPy array, handling quantized tensors and CUDA tensors.
    """
    if isinstance(tensor, np.ndarray):
        return tensor

    if tensor is None:
        return tensor

    if tensor.is_quantized:
        tensor = tensor.dequantize()
    # if tensor is allocated on GPU, first copy to CPU
    # then detach from the current graph and convert to numpy array
    if hasattr(tensor, 'is_cuda'):
        if tensor.is_cuda:
            return tensor.cpu().detach().numpy()

    # if tensor is on CPU only
    if hasattr(tensor, 'detach'):
        return tensor.detach().numpy()

    if hasattr(tensor, 'numpy'):
        return tensor.numpy()

    return np.array(tensor)

def set_model(model: torch.nn.Module, train_mode: bool = False):
    """
    Set model to work in train/eval mode and GPU mode if GPU is available

    Args:
        model: Pytorch model
        train_mode: Whether train mode or eval mode
    Returns:

    """
    if train_mode:
        model.train()
    else:
        model.eval()

    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()