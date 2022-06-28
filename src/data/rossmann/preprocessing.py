import pandas as pd
import numpy as np
import os, pickle
from deprecated import deprecated

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder
from pytorch_forecasting.data.encoders import TorchNormalizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

from data.data_formatters import GenericDataFormatter


@deprecated('Currently not used')
class Rossman:
    def __init__(self, loc='data/raw/rossmann-store-sales/train.csv'):
        self._df = pd.read_csv(loc)
        self._df['Date'] = pd.to_datetime(self._df['Date'])
#         self._df['dow'] = self._df['Date'].dt.dayofweek
        self._df['day'] = self._df['Date'].dt.day
        self._df['month'] = self._df['Date'].dt.month
        self._df['year'] = self._df['Date'].dt.year
        self._df = self._df.drop(columns=['Date'])
        self._df['StateHoliday'] = self._df['StateHoliday'].astype('category').cat.codes

        for col in ['day', 'DayOfWeek', 'month', 'year']:
            df_temp = pd.get_dummies(self._df[col], prefix=col)
            self._df.pop(col)
            self._df = self._df.join(df_temp)

    def get_store(self, store_num):
        return self.df[self.df['Store'] == store_num].reset_index(drop=True)

    @property
    def df(self): return self._df

    @property
    def num_stores(self):
        return len(self._df['Store'].unique())

    def load_sk_dataset(self):
        sample_store_ids = np.random.permutation(self.num_stores)[:100]
        train_x_all, train_y_all, test_x_all, test_y_all = [], [], [], []
        for id in sample_store_ids:
            df = self.get_store(id)
            y = df.pop('Sales').values
            df.pop('Customers')
            X = df.values[:, 1:]

            if len(X) < 30: continue # ignore small samples
            train_x, test_x, train_y, test_y = train_test_split(X, y,
                                                                test_size=self.config['target_size'],
                                                                shuffle=False)
            if self._is_tuning:
                train_x, test_x, train_y, test_y = train_test_split(train_x, train_y,
                                                        test_size=self.config['target_size'],
                                                        shuffle=False)
            train_x_all.append(train_x);train_y_all.append(train_y);
            test_x_all.append(train_x);test_y_all.append(test_y);
            yield {'train_x': train_x, 'train_y': train_y,
                   'test_x': test_x, 'test_y': test_y,
                   'dataset_id': id}

        # train_x, train_y, test_x, test_y = RossmannFormatter._concat_sets(
        #     train_x_all, train_y_all, test_x_all, test_y_all)
        # return {'train_x': train_x, 'train_y': train_y, 'test_x': test_x, 'test_y': test_y}


class RossmannFormatter(GenericDataFormatter):
    """
    Arguments:
    ---
    config (dict)
    """
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

    def _load_dataframe(self,):
        data_dir = "data/raw/rossmann-store-sales/"
        data = pd.read_csv(os.path.join(data_dir, 'train.csv'), parse_dates=['Date'],
            dtype={'Store':str, 'Open':str, 'Promo':str,
                'StateHoliday':str, 'SchoolHoliday':str,
                'DayOfWeek':str, 'Sales':float, 'Customers':float})

        data['StateHoliday'] = data['StateHoliday'].map({'0':0, 'a':1, 'b':2, 'c':3})
        data['StateHoliday'].fillna(0, inplace=True)
        data['StateHoliday'] = data['StateHoliday'].astype(int).astype(str)
        if self._fast_dev_run:
            data = data[data['Store'].isin(data['Store'].unique()[:2])]

        dates = np.sort(data['Date'].unique())
        data['time_idx'] = data['Date'].apply(lambda x: np.where(dates == x)[0][0])

        stores_to_remove = self._get_stores_with_missing_values(data)
        data = data[~data['Store'].isin(stores_to_remove)]
        # data = self._add_lag_features(data, feature_names=['Sales'])
        return data.sort_index(ascending=False)

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
            target='Sales',
            group_ids=['Store'],
            categorical_encoders={},
            min_encoder_length=history_size,
            max_encoder_length=history_size,
            min_prediction_length=target_size,
            max_prediction_length=target_size,
            static_categoricals=['Store'],
            time_varying_known_categoricals=['DayOfWeek', 'Open', 'Promo',
                                             'StateHoliday', 'SchoolHoliday'],
            time_varying_unknown_reals=[],  # 'Sales', 'Customers'
            time_varying_known_reals=['time_idx'],
            target_normalizer=GroupNormalizer(groups=['Store'], center=False),
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
            target='Sales',
            group_ids=['Store'],
            categorical_encoders={},
            min_encoder_length=history_size,
            max_encoder_length=history_size,
            min_prediction_length=target_size,
            max_prediction_length=target_size,
            static_categoricals=['Store'],
            time_varying_known_categoricals=['DayOfWeek', 'Open', 'Promo',
                                             'StateHoliday', 'SchoolHoliday'],
            time_varying_unknown_reals=['Sales'],  # 'Sales', 'Customers'
            time_varying_known_reals=['time_idx'],
            target_normalizer=GroupNormalizer(groups=['Store'], center=False),
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

    @staticmethod
    def _get_stores_with_missing_values(data,):
        loc = 'data/interim/rossmann_bad_stores.pickle'
        if os.path.exists(loc):
            pickle.load(open(loc, 'rb'))

        stores_with_missing_values = []
        for store in data['Store'].unique():
            idxs = data[data['Store'] == store].time_idx.tolist()
            for i in range(len(idxs) - 1):
                temp = idxs[i]
                new = idxs[i + 1]
                if temp - new != 1:
                    # print(store, temp, new)
                    stores_with_missing_values.append(store)
                    break

        pickle.dump(stores_with_missing_values, open(loc, 'wb'))
        return stores_with_missing_values

    # Sklearn methods
    def load_sk_dataset(self,):
        data = self._load_dataframe()
        # data = self._add_lag_features(data, ['Sales', 'Customers'])
        # train_x_all, train_y_all, test_x_all, test_y_all = [], [], [], []
        for store_num in list(data['Store'].unique()):
            df = data[data['Store'] == store_num].reset_index(drop=True)

            y = df.pop('Sales').values
            df.pop('Customers')
            df.pop('Date')

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
                   'dataset_id': store_num}
        #     train_x_all.append(train_x);train_y_all.append(train_y);
        #     test_x_all.append(test_x);test_y_all.append(test_y);

        # train_x, train_y, test_x, test_y = self._concat_sets(train_x_all,\
        #                                     train_y_all, test_x_all, test_y_all)
        # dataset = dict(
        #     train_x=train_x,
        #     train_y=train_y,
        #     test_x=test_x,
        #     test_y=test_y,
        #     config=dict(n_features=train_x.shape[1])
        # )
        # return dataset

    @staticmethod
    @deprecated('Currently not used')
    def _add_lag_features(df, feature_names):
        """ Adds Sales-1 and Customers-1 as features """
        for store_id in df['Store'].unique():
            store_mask = df['Store'] == store_id
            store = df.loc[df['Store'] == store_id]
            for feature in feature_names:
                lag_feature_name = f"{feature}_1"
                df.loc[store_mask, lag_feature_name] = store[feature].shift(1)
        print(len(df))
        df.dropna(inplace=True)
        print(len(df))
        return df

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
