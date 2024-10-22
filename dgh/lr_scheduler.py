from typing import Dict, Any, Optional, List, Tuple, Union

from torch.optim.optimizer import Optimizer
from torch import inf

class ReduceLROnPlateauWithReset:
    """
    Custom learning rate scheduler that reduces the learning rate when a metric has stopped improving,
    with the ability to reset the scheduler state.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        mode: str = 'min',
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = 'rel',
        cooldown: int = 0,
        min_lr: Union[float, List[float], Tuple[float, ...]] = 0,
        eps: float = 1e-8,
        verbose: bool = False
    ) -> None:
        """
        Initializes the ReduceLROnPlateauWithReset scheduler.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            mode (str, optional): One of `min` or `max`. In `min` mode, the learning rate will
                                  be reduced when the quantity monitored has stopped decreasing.
                                  In `max` mode it will be reduced when the quantity monitored has
                                  stopped increasing. Defaults to 'min'.
            factor (float, optional): Factor by which the learning rate will be reduced. new_lr = lr * factor.
                                      Must be < 1.0. Defaults to 0.1.
            patience (int, optional): Number of epochs with no improvement after which learning rate will be reduced.
                                      Defaults to 10.
            threshold (float, optional): Threshold for measuring the new optimum, to only focus on significant changes.
                                         Defaults to 1e-4.
            threshold_mode (str, optional): One of `rel` or `abs`. In `rel` mode, dynamic_threshold = best * (1 + threshold)
                                            in `max` mode or best * (1 - threshold) in `min` mode.
                                            In `abs` mode, dynamic_threshold = best + threshold in `max` mode
                                            or best - threshold in `min` mode. Defaults to 'rel'.
            cooldown (int, optional): Number of epochs to wait before resuming normal operation after lr has been reduced.
                                      Defaults to 0.
            min_lr (float or list or tuple, optional): A scalar or a list of scalars specifying a minimum learning rate
                                                       for each parameter group. Defaults to 0.
            eps (float, optional): Minimal decay applied to lr. If the difference between new and old lr is smaller than eps,
                                   the update is ignored. Defaults to 1e-8.
            verbose (bool, optional): If `True`, prints a message to stdout for each update. Defaults to False.

        Raises:
            ValueError: If `factor` is greater than or equal to 1.0.
            TypeError: If `optimizer` is not an instance of Optimizer.
            ValueError: If `min_lr` is a list, tuple and its length does not match `optimizer.param_groups`.
        """

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(min_lr, (list, tuple)):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = 0
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self) -> None:
        """
        Resets the number of bad epochs and the cooldown counter.
        """
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics: float, epoch: Optional[int] = None) -> None:
        """
        Checks whether the learning rate needs to be updated based on the provided metric.

        Args:
            metrics (float): The metric to monitor.
            epoch (Optional[int], optional): The current epoch number. If not provided, it will be incremented automatically.
                                             Defaults to None.
        """
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
            self.best = self.mode_worse

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def _reduce_lr(self, epoch: int) -> None:
        """
        Reduces the learning rate for each parameter group based on the reduction factor.

        Args:
            epoch (int): The current epoch number.
        """
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    epoch_str = ("%.2f" if isinstance(epoch, float) else
                                 "%.5d") % epoch
                    print('Epoch {}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch_str, i, new_lr))

    @property
    def in_cooldown(self) -> bool:
        """
        Checks if the scheduler is in cooldown mode.

        Returns:
            bool: `True` if in cooldown, `False` otherwise.
        """
        return self.cooldown_counter > 0

    def is_better(self, a: float, best: float) -> bool:
        """
        Determines if the current metric is better than the best recorded metric based on the mode and threshold.

        Args:
            a (float): Current metric value.
            best (float): Best recorded metric value.

        Returns:
            bool: `True` if `a` is better than `best`, `False` otherwise.
        """
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode: str, threshold: float, threshold_mode: str) -> None:
        """
        Initializes the comparison function based on the mode and threshold mode.

        Args:
            mode (str): Mode for comparison ('min' or 'max').
            threshold (float): Threshold for determining improvement.
            threshold_mode (str): Threshold mode ('rel' or 'abs').

        Raises:
            ValueError: If `mode` or `threshold_mode` is invalid.
        """
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    def state_dict(self) -> Dict[str, Any]:
        """
        Returns the state of the scheduler as a dictionary.

        Returns:
            Dict[str, Any]: Scheduler state.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Loads the scheduler state.

        Args:
            state_dict (Dict[str, Any]): Scheduler state. Should be an object returned from a call to `state_dict()`.
        """
        self.__dict__.update(state_dict)
        self._init_is_better(mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode)
