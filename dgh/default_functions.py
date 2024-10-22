from functools import partial
from typing import Callable, Tuple, Any

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from constants import SchedularType, BatchNormAlignemntLossType
from dgh.lr_scheduler import ReduceLROnPlateauWithReset

def reduce_lr_on_platu_step_fn(
    scheduler: ReduceLROnPlateau,
    i_iter: int,
    loss_value: float
) -> None:
    """
    Step function for ReduceLROnPlateau scheduler, using the loss value.

    Args:
        scheduler (ReduceLROnPlateau): The learning rate scheduler.
        i_iter (int): The current iteration.
        loss_value (float): The current loss value.
    """
    scheduler.step(loss_value)

def scheduler_step_fn(
    scheduler: StepLR,
    i_iter: int,
    loss_value: float
) -> None:
    """
    Step function for StepLR scheduler.

    Args:
        scheduler (StepLR): The learning rate scheduler.
        i_iter (int): The current iteration.
        loss_value (float): The current loss value (not used in this scheduler).
    """
    scheduler.step()

def default_bna_loss_fn(
    a: torch.Tensor,
    b: torch.Tensor
) -> torch.Tensor:
    """
    Default BatchNorm alignment loss function, calculates L2 norm squared.

    Args:
        a (torch.Tensor): First input tensor.
        b (torch.Tensor): Second input tensor.

    Returns:
        torch.Tensor: Calculated loss.
    """
    return torch.linalg.norm(a - b) ** 2 / b.size(0)

def get_scheduler_step_fn(
    scheduler_type: str,
    n_iter: int
) -> Tuple[Callable[[Any, int, float], None], Callable[..., Any]]:
    """
    Get the scheduler step function and the corresponding scheduler based on the type.

    Args:
        scheduler_type (str): The type of the scheduler.
        n_iter (int): Number of iterations.

    Returns:
        Tuple[Callable[[Any, int, float], None], Callable[..., Any]]:
            - Scheduler step function.
            - Partial scheduler constructor.
    """
    if scheduler_type is None or scheduler_type == SchedularType.REDUCE_ON_PLATEAU.value:
        scheduler = partial(ReduceLROnPlateau, min_lr=1e-4, factor=0.5, patience=int(n_iter / 50))
        return reduce_lr_on_platu_step_fn, scheduler
    if scheduler_type is None or scheduler_type == SchedularType.REDUCE_ON_PLATEAU_WITH_RESET.value:
        scheduler = partial(ReduceLROnPlateauWithReset, min_lr=1e-4, factor=0.5, patience=int(n_iter / 50))
        return reduce_lr_on_platu_step_fn, scheduler
    elif scheduler_type == SchedularType.CONST.value:
        scheduler = partial(StepLR, step_size=int(n_iter), gamma=0.5)
        return scheduler_step_fn, scheduler
    else:
        scheduler = partial(StepLR, step_size=int(n_iter / 15), gamma=0.5)
        return scheduler_step_fn, scheduler

def get_bn_loss_fn(
    bna_loss_type: str
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Get the BatchNorm alignment loss function based on the type.

    Args:
        bna_loss_type (str): The type of BatchNorm alignment loss.

    Returns:
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]: The loss function.

    Raises:
        Exception: If an unknown batch normalization alignment loss type is specified.
    """
    if bna_loss_type is None or bna_loss_type == BatchNormAlignemntLossType.L2_SQUARE.value:
        return default_bna_loss_fn
    else:
        raise Exception(f'Unknown "layer_weighting_type" {bna_loss_type}. Please choose one of {BatchNormAlignemntLossType.get_values()}')
