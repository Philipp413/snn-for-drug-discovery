import json
import os
from abc import ABC
from typing import Callable, Dict, List

import numpy as np

_SAVE_FORMAT = "{basedir}/epoch-{epoch_ix:03d}"

class TorchCallback(ABC):
    """Base class for all Torch callbacks."""

    def __init__(self) -> None:
        """Creates a TorchCallback. Sets the `stop_training` flag to `False`, which would be common attribute of all callbacks."""
        super().__init__()
        self.stop_training = False

    def on_epoch_end(self, epoch_ix, history, **kwargs):
        """Called at the end of an epoch.

        Parameters
        ----------
        epoch_ix : int
            The index of the epoch that just ended.
        history : Dict[str, List[float]]
            A dictionary containing the training history. The keys are the names of the metrics, and the values are lists of the metric values at each epoch.
        **kwargs
            Any additional keyword arguments.
        """
        pass

    def on_train_end(self, epoch_ix, history, **kwargs):
        """Called at the end of training.

        Parameters
        ----------
        epoch_ix : int
            The index of the epoch that just ended.
        history : Dict[str, List[float]]
            A dictionary containing the training history. The keys are the names of the metrics, and the values are lists of the metric values at each epoch.
        **kwargs
            Any additional keyword arguments.
        """
        pass

class EarlyStopping(TorchCallback):
    """A callback that stops training when a monitored metric has stopped improving."""

    def __init__(self, patience: int, delta: float, criterion: str, mode: str) -> None:
        """Creates an `EarlyStopping` callback.

        Parameters
        ----------
        patience : int
            Number of epochs to wait for improvement before stopping the training.
        delta : float
            Minimum change in the monitored quantity to qualify as an improvement.
        criterion : str
            The name of the metric to monitor.
        mode : str
            One of `"min"` or `"max"`. In `"min"` mode, training will stop when the quantity monitored has stopped decreasing;
            in `"max"` mode it will stop when the quantity monitored has stopped increasing.
        """
        super().__init__()
        self.patience = patience
        self.delta = delta
        self.criterion = criterion
        if mode not in ["min", "max"]:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
        self.mode = mode
        self.best = np.inf if mode == "min" else -np.inf
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch_ix: int, history: Dict[str, float], **kwargs) -> None:
        """Called at the end of an epoch. Updates the best metric value and the number of epochs waited for improvement.
        `stop_training` attribute is set to `True` if the training should be stopped.

        Parameters
        ----------
        epoch_ix : int
            The index of the epoch that just ended.
        history : Dict[str, float]
            A dictionary containing the training history. The keys are the names of the metrics, and the values are lists of the metric values at each epoch.
        """
        monitor_values = history[self.criterion]
        self.wait += 1
        print(f"Waiting for: {self.wait}")
        if len(monitor_values) < self.patience:
            return

        current = monitor_values[epoch_ix]
        if self._is_improvement(current):
            self.best = current
            self.best_epoch = epoch_ix
            self.wait = 0
        elif self.wait >= self.patience:
            self.stop_training = True
            print("STOP training")
            self.stopped_epoch = epoch_ix

    def _is_improvement(self, current):
        if self.mode == "min":
            return current < self.best - self.delta

        return current > self.best + self.delta