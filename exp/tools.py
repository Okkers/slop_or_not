import os
import random

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    """
    Keeps the checkpoint with the best validation score and stops after `patience` epochs
    without improvement.

    metric = 'loss' (default, lower is better) or 'f1' / 'auc' / 'accuracy' (higher is better).
    Note: selecting on F1 alone is risky on balanced data - a model that predicts "machine"
    for everything already has F1 = 0.67 - so validation loss is the default and F1 is
    what we REPORT on the test set.
    """

    def __init__(self, patience: int = 25, metric: str = "loss", verbose: bool = True):
        self.patience = patience
        self.metric = metric
        self.verbose = verbose
        self.counter = 0
        self.best = None
        self.best_epoch = -1
        self.early_stop = False

    def _score(self, val_loss, val_metrics):
        if self.metric == "loss":
            return (-val_loss,)
        return (round(val_metrics[self.metric], 6), -val_loss)   # ties broken by lower loss

    def __call__(self, val_loss, val_metrics, model, path, epoch):
        score = self._score(val_loss, val_metrics)
        if self.best is None or score > self.best:
            self.best = score
            self.best_epoch = epoch
            os.makedirs(path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(path, "checkpoint.pth"))
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
