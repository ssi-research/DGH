from typing import Callable, Optional

import torch
from constants import DEVICE
from model_evaluation.classification_accuracy import model_accuracy_evaluation


class ModelParams(object):
    """
   Represents parameters and configurations for a model.
   """

    def __init__(
            self,
            model: torch.nn.Module,
            float_accuracy: float,
            evaluation_fn: Callable,
            img_size: int,
            batch_size: int = 50,
            max_batch_for_data_gen: int = 64
    ) -> None:
        """
        Initializes the ModelParams with the given parameters.

        Args:
            model (torch.nn.Module): The PyTorch model.
            float_accuracy (float): The accuracy of the model in floating point.
            evaluation_fn (Callable): The function to evaluate the model's accuracy.
            img_size (int): The size of the input images.
            batch_size (int, optional): The batch size for validation. Defaults to 50.
            max_batch_for_data_gen (int, optional): The maximum batch size for data generation. Defaults to 64.
        """
        self.model = model
        self.float_accuracy = float_accuracy
        self.evaluation_fn = evaluation_fn
        self.validation_batch_size = batch_size
        self.img_size = img_size
        self.batch_size = batch_size
        self.max_batch_for_data_gen = max_batch_for_data_gen

    def get_model(self) -> torch.nn.Module:
        """
        Prepares and returns the model for evaluation.

        Returns:
            torch.nn.Module: The prepared model.
        """
        self.model.eval()
        self.model.to(DEVICE)
        return self.model

    def get_batch_size(self) -> int:
        """
        Retrieves the batch size for validation.

        Returns:
            int: The batch size.
        """
        return self.batch_size

    def get_img_size(self, override_size: Optional[int] = None) -> int:
        """
        Retrieves the image size, optionally overriding it.

        Args:
            override_size (Optional[int], optional): The size to override the default image size. Defaults to None.

        Returns:
            int: The image size.
        """
        if override_size is not None:
            return override_size
        return self.img_size


class PytorchModelParams(ModelParams):
    """
   Represents parameters and configurations specific to PyTorch models.
   """
    def __init__(
            self,
            model: torch.nn.Module,
            float_accuracy: float,
            img_size: int,
            batch_size: int = 50,
            max_batch_for_data_gen: int = 64
    ) -> None:
        """
        Initializes the PytorchModelParams with the given parameters.

        Args:
            model (torch.nn.Module): The PyTorch model.
            float_accuracy (float): The accuracy of the model in floating point.
            img_size (int): The size of the input images.
            batch_size (int, optional): The batch size for validation. Defaults to 50.
            max_batch_for_data_gen (int, optional): The maximum batch size for data generation. Defaults to 64.
        """
        evaluation_fn = model_accuracy_evaluation
        super(PytorchModelParams, self).__init__(model, float_accuracy, evaluation_fn, img_size, batch_size, max_batch_for_data_gen)

