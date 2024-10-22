from typing import Tuple, List

import numpy as np
import progressbar
import torch
from constants import DEVICE


def eval(outputs: torch.Tensor, labels: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> torch.Tensor:
    """
    Evaluates the network outputs against the ground-truth labels for the specified top-k predictions.

    Args:
        outputs (torch.Tensor): The output logits from the model.
        labels (torch.Tensor): The ground-truth labels.
        topk (Tuple[int, ...], optional): Tuple of top-k values to consider for evaluation. Defaults to (1,).

    Returns:
        torch.Tensor: A tensor indicating correct predictions.
    """
    maxk = max(topk)

    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    return correct


def accuracy(outputs: torch.Tensor, labels: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Computes the accuracy over the k top predictions for the specified values of k.

    Args:
        outputs (torch.Tensor): The output logits from the model.
        labels (torch.Tensor): The ground-truth labels.
        topk (Tuple[int, ...], optional): Tuple of top-k values to consider for accuracy. Defaults to (1,).

    Returns:
        Tuple[List[torch.Tensor], torch.Tensor]:
            - A list of accuracies for each k in topk.
            - A tensor indicating correct predictions.
    """
    correct = eval(outputs, labels)

    batch_size = labels.size(0)

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res, correct


def model_accuracy_evaluation(model: torch.nn.Module, val_data_loader: torch.utils.data.DataLoader) -> Tuple[float, np.ndarray]:
    """
    Evaluates the model's accuracy on the validation dataset.

    The evaluation is performed over the top-1 and top-5 predictions.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        val_data_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.

    Returns:
        Tuple[float, np.ndarray]:
            - The top-1 accuracy as a float.
            - An array of correct prediction indicators.
    """
    acc_top1 = 0
    acc_top5 = 0

    model = model.to(DEVICE)
    model = model.eval()

    batch_cntr = 1
    iterations = len(val_data_loader)
    correct_inds_list = []
    with progressbar.ProgressBar(max_value=len(val_data_loader)) as progress_bar:
        with torch.no_grad():

            for input_data, target_data in val_data_loader:
                inputs_batch = input_data.to(DEVICE)
                target_batch = target_data.to(DEVICE)

                predicted_batch = model(inputs_batch)

                batch_avg_top_1_5, correct_inds = accuracy(outputs=predicted_batch, labels=target_batch,
                                             topk=(1, 5))
                correct_inds_list.append(correct_inds)
                acc_top1 += batch_avg_top_1_5[0].item()
                acc_top5 += batch_avg_top_1_5[1].item()

                progress_bar.update(batch_cntr)

                batch_cntr += 1
                if batch_cntr > iterations:
                    break

    acc_top1 /= iterations
    acc_top5 /= iterations

    correct_inds = torch.cat(correct_inds_list, 1).cpu().detach().numpy()
    return acc_top1, correct_inds