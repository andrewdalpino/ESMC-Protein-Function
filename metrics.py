import torch

from torch import Tensor

from torch.nn import Module

from torchmetrics.classification import BinaryPrecision, BinaryRecall


class F1Score(Module):
    """Computes the F1 score."""

    def __init__(self):
        super().__init__()

        self.precision_metric = BinaryPrecision()
        self.recall_metric = BinaryRecall()

    def update(self, y_pred: Tensor, y: Tensor) -> None:
        self.precision_metric.update(y_pred, y)
        self.recall_metric.update(y_pred, y)

    def compute(self) -> tuple[Tensor, Tensor, Tensor]:
        precision = self.precision_metric.compute()
        recall = self.recall_metric.compute()

        if precision + recall == 0:
            f1_score = torch.tensor(0.0, device=precision.device)
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)

        return f1_score, precision, recall

    def reset(self) -> None:
        self.precision_metric.reset()
        self.recall_metric.reset()
