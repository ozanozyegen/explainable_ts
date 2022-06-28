import pandas as pd
import numpy as np
import scipy as sp
import sys, os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from scipy import stats
from datetime import datetime
from data.helpers import week_of_month

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder
from pytorch_forecasting.data.encoders import TorchNormalizer
from data.data_formatters import GenericDataFormatter


class WalmartFormatter(GenericDataFormatter):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def _load_dataframe(self, data_path='data/processed/walmart.csv'):
        if not os.path.exists(data_path):
            import data.walmart.raw_data_process
        data = pd.read_csv(data_path, parse_dates=['Date'],
            dtype={'Weekly_Sales': np.float32, 'Store': str, 'Dept': str,
                   'year': str, 'month': str, 'weekofmonth': str, 'day': str,
                   'Type': str, 'Size': int, 'Temperature': np.float32,
                   'Fuel_Price': np.float32, 'CPI': np.float32,
                   'Unemployment': np.float32,
                   'IsHoliday': str, 'time_idx': int})
        if self._fast_dev_run:
            data = data[data['Store'].isin(data['Store'].unique()[:2])]
        return data

    def load_sk_dataset(self,):
        data = self._load_dataframe().sort_index(ascending=True)

        # train_x_all, train_y_all, test_x_all, test_y_all = [], [], [], []
        for selected_store_dept in tqdm(data[['Store', 'Dept']].drop_duplicates().values):
            df = data[(data['Store'] == selected_store_dept[0]) &
                      (data['Dept'] == selected_store_dept[1])]
            y = df.pop('Weekly_Sales')
            df.pop('Unnamed: 0');df.pop('Date');
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
                   'dataset_id': ','.join(selected_store_dept.tolist())}

        #     train_x_all.append(train_x);train_y_all.append(train_y);
        #     test_x_all.append(test_x);test_y_all.append(test_y);

        # train_x, train_y, test_x, test_y = self._concat_sets(train_x_all,\
        #                                     train_y_all, test_x_all, test_y_all)
        # dataset = dict(
        #     train_x=train_x,
        #     train_y=train_y,
        #     test_x=test_x,
        #     test_y=test_y,
        #     config=dict(n_features=train_x.shape[1],
        #                 features=list(df.columns))
        # )
        # return dataset

    @staticmethod
    def _concat_sets(train_x_all, train_y_all, test_x_all, test_y_all):
        concat = lambda x: np.concatenate(x, axis=0)
        print('Packing dataset')
        train_x = concat(train_x_all)
        train_y = concat(train_y_all)
        test_x = concat(test_x_all)
        test_y = concat(test_y_all)
        print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
        return train_x, train_y, test_x, test_y

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
            target='Weekly_Sales',
            group_ids=['Store', 'Dept'],
            categorical_encoders={},
            min_encoder_length=history_size,
            max_encoder_length=history_size,
            min_prediction_length=target_size,
            max_prediction_length=target_size,
            static_categoricals=['Store', 'Dept', 'Type'],
            static_reals=['Size'],
            time_varying_known_categoricals=['year', 'month', 'weekofmonth',
                                             'day', 'IsHoliday'],
            time_varying_unknown_reals=[],  # 'Weekly_Sales'
            time_varying_known_reals=['time_idx',
                                      'Temperature', 'Fuel_Price',
                                      'CPI', 'Unemployment'],
            target_normalizer=GroupNormalizer(groups=['Store', 'Dept'], center=False),
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
            target='Weekly_Sales',
            group_ids=['Store', 'Dept'],
            categorical_encoders={},
            min_encoder_length=history_size,
            max_encoder_length=history_size,
            min_prediction_length=target_size,
            max_prediction_length=target_size,
            static_categoricals=['Store', 'Dept', 'Type'],
            static_reals=['Size'],
            time_varying_known_categoricals=['year', 'month', 'weekofmonth',
                                             'day', 'IsHoliday'],
            time_varying_unknown_reals=['Weekly_Sales'],
            time_varying_known_reals=['time_idx',
                                      'Temperature', 'Fuel_Price',
                                      'CPI', 'Unemployment'],
            target_normalizer=GroupNormalizer(groups=['Store', 'Dept'], center=False),
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