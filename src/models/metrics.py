import numpy as np
from typing import List
import torch
from pytorch_forecasting.metrics import MultiHorizonMetric


def _deviation_across_batches(loss, y_pred, target, per_batch=32):
    losses = []
    print(y_pred.shape, target.shape)
    for idx in range(y_pred.shape[0] // per_batch):
        batch_start = idx * per_batch
        losses.append(loss(y_pred[batch_start: batch_start + 32],
                           target[batch_start: batch_start + 32]))
    losses = torch.Tensor(losses)
    return losses.std()


def nrmse_np(y_pred, target):
    def _nrmse(y_pred, target):
        mse = np.mean((y_pred - target)**2)
        rmse = np.sqrt(mse)
        denominator = np.mean(target)
        return rmse / denominator
    nrmse = _nrmse(y_pred, target)
    std = _deviation_across_batches(_nrmse, y_pred, target)
    return nrmse, std


class NRMSE(MultiHorizonMetric):
    """
    Normalized root mean squared error

    Defined as ``RMSE / target.mean()``
    """
    def __init__(self, reduction: str = 'none', **kwargs) -> None:
        super().__init__(reduction, **kwargs)

    def loss(self, y_pred, target):
        mse = torch.mean(torch.pow(self.to_prediction(y_pred) - target, 2))
        rmse = torch.sqrt(mse)
        denominator = torch.mean(target)
        return rmse / denominator

    def loss2(self, y_pred, target):
        """ Ignores 0 idxs """
        y_pred2 = y_pred.clone()
        y_pred2[target == 0] = 0
        return self.loss(y_pred2, target)

    def loss_deviation(self, y_pred, target):
        return _deviation_across_batches(self.loss, y_pred, target)


def nd_np(y_pred, target):
    def _nd(y_pred, target):
        nominator = np.sum(np.abs(y_pred - target))
        denominator = np.sum(np.abs(target))
        return nominator / denominator
    nd = _nd(y_pred, target)
    std = _deviation_across_batches(_nd, y_pred, target)
    return nd, std


class ND(MultiHorizonMetric):
    """
    Normalized Deviation
    """
    def __init__(self, reduction: str = 'none', **kwargs) -> None:
        super().__init__(reduction, **kwargs)

    def loss(self, y_pred, target):
        nominator = torch.sum(torch.abs(self.to_prediction(y_pred) - target))
        denominator = torch.sum(torch.abs(target))
        return nominator / denominator

    def loss2(self, y_pred, target):
        """ Ignores 0 idxs """
        y_pred2 = y_pred.clone()
        y_pred2[target == 0] = 0
        return self.loss(y_pred2, target)

    def loss_deviation(self, y_pred, target):
        return _deviation_across_batches(self.loss, y_pred, target)


def rho_risk_np(y_pred, target, quantiles=[0.5, 0.75, 0.9]):
    losses = []
    for i, q in enumerate(quantiles):
        over_pred = target < y_pred[..., i]
        under_pred = target >= y_pred[..., i]
        numerator_weights = q * over_pred + (q - 1) * under_pred
        numerator = 2 * (y_pred[..., i] - target) * numerator_weights
        loss = np.sum(numerator) / np.sum(target)
        print(loss)
        losses.append(loss)
    return losses


class RhoRisk(MultiHorizonMetric):
    """
    Rho Risk is defined as Normalized Quantile Loss

    See @salinas2020deepar for the definition
    """

    def __init__(
        self,
        quantiles: List[float] = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
        reduction: str = 'none',
        **kwargs,
    ):
        """
        Quantile loss

        Args:
            quantiles: quantiles for metric
        """
        super().__init__(reduction, quantiles=quantiles, **kwargs)

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # calculate quantile loss
        losses = []
        for i, q in enumerate(self.quantiles):
            over_pred = target < y_pred[..., i]
            under_pred = target >= y_pred[..., i]
            numerator_weights = q * over_pred + (q - 1) * under_pred
            numerator = 2 * (y_pred[..., i] - target) * numerator_weights
            loss = torch.sum(numerator) / torch.sum(target)
            print(loss)
            losses.append(loss)
            # losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
        # losses = torch.cat(losses, dim=2)
        return losses

    def loss2(self, y_pred, target):
        """ Ignores 0 idxs """
        y_pred2 = y_pred.clone()
        y_pred2[target == 0] = 0
        return self.loss(y_pred2, target)

    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert network prediction into a point prediction.

        Args:
            y_pred: prediction output of network

        Returns:
            torch.Tensor: point prediction
        """
        if y_pred.ndim == 3:
            idx = self.quantiles.index(0.5)
            y_pred = y_pred[..., idx]
        return y_pred

    def to_quantiles(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert network prediction into a quantile prediction.

        Args:
            y_pred: prediction output of network

        Returns:
            torch.Tensor: prediction quantiles
        """
        return y_pred