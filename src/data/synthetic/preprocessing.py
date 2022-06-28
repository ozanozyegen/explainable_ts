import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import TorchNormalizer
from sklearn.model_selection import train_test_split

from data.data_formatters import GenericDataFormatter
from data.synthetic.create_samples import create_dataset_feat_ranking


class SyntheticFormatter(GenericDataFormatter):
    def __init__(self, config: dict):
        super().__init__(config)

    def _load_dataframe(self):
        if self._fast_dev_run:
            data = pd.DataFrame(create_dataset_feat_ranking(total_length=50))
        else:
            data = pd.DataFrame(create_dataset_feat_ranking())
        return data

    def load_sk_dataset(self,):
        data = self._load_dataframe()
        for series_id in list(data['series_id'].unique()):
            df = data[data['series_id'] == series_id].reset_index(drop=True)
            y = df.pop('y')
            df.drop(['time_idx', 'series_id'], axis=1, inplace=True)
            X = df.values
            train_x, test_x, train_y, test_y = train_test_split(X, y,
                                                test_size=self.config['target_size'],
                                                shuffle=False)
            if self._is_tuning:
                train_x, test_x, train_y, test_y = train_test_split(train_x, train_y,
                                                test_size=self.config['target_size'],
                                                shuffle=False)
            yield {'train_x': train_x, 'train_y': train_y,
                    'test_x': test_x, 'test_y': test_y,
                    'dataset_id': series_id}

    def _get_dense_loaders(self, data):
        batch_size = self.config['batch_size']
        history_size, target_size = self.config['history_size'], self.config['target_size']

        if self._is_tuning:
            training_cutoff = data['time_idx'].max() - target_size
        else:
            training_cutoff = data['time_idx'].max()

        training = TimeSeriesDataSet(
            data[lambda x: x.time_idx <= training_cutoff],
            time_idx='time_idx',
            target='y',
            group_ids=['series_id'],
            categorical_encoders={},
            min_encoder_length=history_size,
            max_encoder_length=history_size,
            min_prediction_length=target_size,
            max_prediction_length=target_size,
            static_categoricals=[],
            static_reals=[],
            time_varying_known_categoricals=['c1'],
            time_varying_unknown_reals=[],
            time_varying_known_reals=['x1', 'x2'],
            target_normalizer=TorchNormalizer(center=False),
            # center false puts range to 0 - 3.7 for interpretability score calculation
            add_relative_time_idx=self.config['add_relative_time_idx'],
            add_target_scales=self.config['add_target_scales'],
            randomize_length=None,
        )

        validation = TimeSeriesDataSet.from_dataset(training, data, predict=True,
                                                    stop_randomization=True)
        train_dataloader = training.to_dataloader(train=True,
                                    batch_size=batch_size, num_workers=0)
        val_dataloader = validation.to_dataloader(train=False,
                                    batch_size=batch_size * 10, num_workers=0)
        return training, train_dataloader, val_dataloader

    def _get_autoregressive_loaders(self, data):
        batch_size = self.config['batch_size']
        history_size, target_size = self.config['history_size'], self.config['target_size']

        if self._is_tuning:
            training_cutoff = data['time_idx'].max() - target_size
        else:
            training_cutoff = data['time_idx'].max()

        training = TimeSeriesDataSet(
            data[lambda x: x.time_idx <= training_cutoff],
            time_idx='time_idx',
            target='y',
            group_ids=['series_id'],
            categorical_encoders={},
            min_encoder_length=history_size,
            max_encoder_length=history_size,
            min_prediction_length=target_size,
            max_prediction_length=target_size,
            static_categoricals=[],
            static_reals=[],
            time_varying_known_categoricals=['c1'],
            time_varying_unknown_reals=['y'],
            time_varying_known_reals=['x1', 'x2'],
            target_normalizer=TorchNormalizer(center=False),
            # center false puts range to 0 - 3.7 for interpretability score calculation
            add_relative_time_idx=self.config['add_relative_time_idx'],
            add_target_scales=self.config['add_target_scales'],
            randomize_length=None,
        )

        validation = TimeSeriesDataSet.from_dataset(training, data, predict=True,
                                                    stop_randomization=True)
        train_dataloader = training.to_dataloader(train=True,
                                    batch_size=batch_size, num_workers=0)
        val_dataloader = validation.to_dataloader(train=False,
                                    batch_size=batch_size * 10, num_workers=0)
        return training, train_dataloader, val_dataloader
