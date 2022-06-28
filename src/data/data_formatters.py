import abc
import numpy as np


class GenericDataFormatter(abc.ABC):
    """Abstract base class for all data formatters.

    User can implement the abstract methods below to perform dataset-manipulations
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self._scalers = dict()
        self._fast_dev_run = config.get('fast_dev_run', False)
        self._is_tuning = config.get('is_tuning', False)

    @staticmethod
    def _set_category_sizes(train_x: np.ndarray, cat_feat_idxs):
        """ Sets the category sizes for the categorical feature embeddings
        """
        category_sizes = []
        for cat_feat_idx in cat_feat_idxs:
            train_cat = train_x[:, :, cat_feat_idx]
            category_sizes.append(int(np.max(train_cat) + 1))
        return category_sizes

    def load_dataset(self, ):
        data = self._load_dataframe()
        model_name = self.config['MODEL']
        if model_name in ['mlp', 'emb_net', 'polar_dense']:
            return self._get_dense_loaders(data)
        elif model_name in ['polar_rnn', 'rnn']:
            return self._get_autoregressive_loaders(data)

