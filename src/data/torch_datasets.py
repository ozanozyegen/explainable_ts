from lib2to3.pytree import Base
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm


def _get_scale_factor(config, series):
    def _deepar_scaling(series):
        """Expects 2D input (num_series, history_size)"""
        return 1 + series.mean(axis=1)

    def _abs_scaling(series):
        """Expects 2D input (num_series, history_size)
            Useful when predicting the difference, or
            your input contains negative vals
        """
        return 10 * np.abs(series).max(axis=1) + 1

    scale_factor = config.get('scale_factor', None)
    if scale_factor is None:
        return np.ones((series.shape[0], 1))
    elif scale_factor == 'deepar':
        return np.expand_dims(_deepar_scaling(series), axis=-1)
    elif scale_factor == 'abs':
        return np.expand_dims(_abs_scaling(series), axis=-1)


class BaseDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        pass

class SimpleDataset(BaseDataset):
    def __init__(self, config, X, y):
        super().__init__(config)
        self.X, self.y = X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

class TorchDataset(BaseDataset):
    """ Provide data in formats suitable to all ts forecasting models
    Returns:
        - ar_inp: (batch_size, history_size, features)
        - v: Normalization constant to scale and descale
        - ar_label: Labels for the features
        - data: Full dataset for monte-carlo sampling
    """
    def __init__(self, config, dataset, split):
        super().__init__(config)
        history_size = config['history_size']
        if split == 'train':
            self.data = dataset['train_x']
            self.v = _get_scale_factor(config, self.data[:, :, 0])
            self.data[:, :, 0] = self.data[:, :, 0] / self.v  # Scale
            train_target = self.data[:, :history_size, 0]
            self.label = self.data[:, history_size:, 0]
        elif split == 'test':
            self.data = dataset['test_x']
            self.v = _get_scale_factor(config, self.data[:, :, 0])  # (batch_size,)
            self.data[:, :, 0] = self.data[:, :, 0] / self.v  # Scale
            train_target = self.data[:, :history_size, 0]
            self.label = self.data[:, history_size:, 0]

        self.ar_inp = self.data[:, :history_size, :]
        # Autoregressive (one-step ahead) label
        self.ar_label = np.concatenate((train_target[:, 1:], self.label[:, 0:1]), axis=1)
        self.ar_label = self.ar_label[:, :, np.newaxis]

    def __getitem__(self, index):
        return (self.ar_inp[index], self.v[index], self.ar_label[index],
                self.data[index], self.label[index])

class WeightedSampler(Sampler):
    def __init__(self, config, dataset, replacement=True):
        self.data = dataset['train_x']
        train_sales = self.data[:, config['history_size'], 0]
        v = _get_scale_factor(config, train_sales)
        self.weights = torch.as_tensor(np.abs(v)/np.sum(np.abs(v)), dtype=torch.double)
        self.num_samples = self.weights.shape[0]
        self.replacement = replacement

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, self.replacement).tolist())

    def __len__(self):
        return self.num_samples
