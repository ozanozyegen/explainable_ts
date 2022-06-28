import os
from pytorch_forecasting import RecurrentNetwork
import wandb
import pickle
from abc import ABC, abstractmethod
from deprecated import deprecated
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelSummary, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder
from pytorch_forecasting.metrics import QuantileLoss, NormalDistributionLoss
from pytorch_forecasting.metrics import SMAPE, RMSE, MAE, MASE, MAPE
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from pytorch_forecasting.models import DecoderMLP
from models.polar_dense import MyDecoderMLP

from configs.defaults import Globs
from data.loader import data_loader
from models.metrics import NRMSE, ND, RhoRisk, nd_np, rho_risk_np, nrmse_np
from models.deepar import DeepAR
from models.pidits import PIDITS
from models.polar_rnn import PolarRNN


def select_trainer(model_name):
    if model_name in ['mlp']:
        return DecoderMLPTrainer
    elif model_name in ['polar_dense']:
        return PolarDenseTrainer
    elif model_name in ['polar_rnn']:
        return PolarRNNTrainer
    elif model_name in ['rnn']:
        return RNNTrainer
    else:
        raise ValueError("Unknown model name")


class Trainer:
    def __init__(self, config: dict, logger=None):
        self.config = config
        self.logger = logger
        if logger is not None:
            self.local_save_dir = f'models/{wandb.run.id}'

    @abstractmethod
    def train(self,):
        pass

    def _save_outputs(self, pred_y, test_y):
        if self.config.get('save_model', True):
            pickle.dump(pred_y,
                open(os.path.join(self.local_save_dir, 'pred_y.pickle'), 'wb'))
            pickle.dump(test_y,
                open(os.path.join(self.local_save_dir, 'test_y.pickle'), 'wb'))


class SklearnTrainer(Trainer):
    def __init__(self, config: dict, logger=None):
        super().__init__(config, logger)

    def build_model(self,):
        raise NotImplementedError

    def train(self, dataset):
        train_x, train_y = dataset['train_x'], dataset['train_y']

        self.model.fit(train_x, train_y)
        val_metrics, pred_package = self.evaluate(dataset)
        if self.config.get('save_model', True):
            if 'dataset_id' in dataset:
                # If separate model trained for each time series
                save_dir = os.path.join(self.local_save_dir, 'models')
                os.makedirs(save_dir, exist_ok=True)
                pickle.dump(self.model,
                    open(os.path.join(save_dir,
                    f"{dataset['dataset_id']}_model.ckpt"), 'wb'))
            else:  # If single model trained for all dataset
                pickle.dump(self.model,
                    open(os.path.join(self.local_save_dir, 'model.ckpt'), 'wb'))
        return val_metrics, pred_package

    def evaluate(self, dataset, prefix='final'):
        test_x, test_y = dataset['test_x'], dataset['test_y']
        pred_y = self.model.predict(test_x)

        val_metrics = self.evaluate_metrics(pred_y, test_y)
        return val_metrics, pred_y

    @staticmethod
    def evaluate_metrics(pred_y, test_y, pred_q=None, quants=None,
                         prefix='final', suffix=''):
        val_metrics = dict()
        nrmse = nrmse_np(pred_y, test_y)
        nd = nd_np(pred_y, test_y)
        dec_5 = lambda x: np.around(x, decimals=5)
        val_metrics[f'{prefix}_NRMSE{suffix}'] = dec_5(nrmse[0].item())
        val_metrics[f'{prefix}_ND{suffix}'] = dec_5(nd[0].item())
        val_metrics[f'{prefix}_NRMSE_std'] = dec_5(nrmse[1].item())
        val_metrics[f'{prefix}_ND_std'] = dec_5(nd[1].item())
        if pred_q is not None and quants is not None:
            losses = rho_risk_np(pred_q, test_y, quants)
            for quant, loss in zip(quants, losses):
                val_metrics[f'{prefix}rho{suffix}-{quant}'] = dec_5(loss.item())
        return val_metrics


class GBRTrainer(SklearnTrainer):
    def build_model(self,):
        self.model = GradientBoostingRegressor(
            n_estimators=self.config['n_estimators'],  # Num trees
            max_depth=self.config['max_depth'],
            min_samples_split=self.config['min_samples_split']
        )

    def evaluate(self, dataset):
        val_metrics, pred_y = super().evaluate(dataset)
        train_x, train_y = dataset['train_x'], dataset['train_y']
        test_x, test_y = dataset['test_x'], dataset['test_y']

        pred_q = []
        quants = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        for i, q in enumerate(quants):
            self.model.set_params(loss='quantile', alpha=q)
            self.model.fit(train_x, train_y)
            pred_q.append(self.model.predict(test_x))
        pred_q = np.array(pred_q).T
        losses = rho_risk_np(pred_q, test_y, quants)
        for quant, loss in zip(quants, losses):
            val_metrics[f'finalrho-{quant}'] = loss
        return val_metrics, (pred_y, test_y, pred_q, quants)


class TorchForecastTrainer(Trainer):
    def __init__(self, config: dict, logger=None):
        super().__init__(config, logger)

    def build_model(self, training):
        raise NotImplementedError()

    def load_from_checkpoint(self, path):
        if not hasattr(self, 'model'):
            raise RuntimeError("Build the model before loading checkpoint!")
        self.model = self.model.load_from_checkpoint(path)

    def get_predictions(self, val_dataloader):
        actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
        predictions = self.model.predict(val_dataloader)
        return actuals, predictions

    def _get_callbacks(self, val_loader):
        early_stop_callback = EarlyStopping(monitor="val_loss",
                        min_delta=1e-4, patience=10, verbose=False, mode="min")
        lr_logger = LearningRateMonitor()  # log the learning rate
        callbacks = [early_stop_callback, lr_logger]
        if self.config.get('save_model', True):
            callbacks.append(ModelCheckpoint(dirpath=self.local_save_dir,
                                             filename='model',
                                             monitor='val_loss', mode='min'))
        return callbacks

    def _get_logging_metrics(self,):
        return nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE(),
                              NRMSE(), ND()])

    def evaluate(self, val_loader, prefix='final') -> dict:
        metrics = {}
        actuals = torch.cat([y[0] for x, y in iter(val_loader)])
        pred = self.model.predict(val_loader)
        self._save_outputs(pred, actuals)
        nd, nrmse = ND().loss(pred, actuals), NRMSE().loss(pred, actuals)
        nd2, nrmse2 = ND().loss2(pred, actuals), NRMSE().loss2(pred, actuals)
        nd_std = ND().loss_deviation(pred, actuals)
        nrmse_std = NRMSE().loss_deviation(pred, actuals)
        # Save metrics
        dec_5 = lambda x: np.around(x, decimals=5)
        metrics[f"{prefix}_ND"] = dec_5(nd.cpu().detach().numpy().item())
        metrics[f"{prefix}_NRMSE"] = dec_5(nrmse.cpu().detach().numpy().item())

        metrics[f"{prefix}_ND2"] = dec_5(nd2.cpu().detach().numpy().item())
        metrics[f"{prefix}_NRMSE2"] = dec_5(nrmse2.cpu().detach().numpy().item())

        metrics[f"{prefix}_ND_std"] = dec_5(nd_std.cpu().detach().numpy().item())
        metrics[f"{prefix}_NRMSE_std"] = dec_5(nrmse_std.cpu().detach().numpy().item())
        return metrics


class DeepARTrainer(TorchForecastTrainer):
    def __init__(self, config: dict, logger=None):
        super().__init__(config, logger)

    def build_model(self, training):
        self.model = DeepAR.from_dataset(
            training,
            learning_rate=self.config['lr'],
            hidden_size=self.config['hidden_size'],
            rnn_layers=self.config['rnn_layers'],
            dropout=self.config['dropout'],
            loss=NormalDistributionLoss(),
            log_val_interval=3,
            # reduce_on_plateau_patience=4,
            logging_metrics=self._get_logging_metrics()
        )
        print(f"Number of parameters in the network: {self.model.size()/1e3:.1f}")

    def train(self, train_loader, val_loader):
        self.trainer = pl.Trainer(
            logger=self.logger,
            # accelerator='cpu',
            gpus=1,
            auto_select_gpus=True,
            gradient_clip_val=0.1,
            auto_lr_find=True,
            limit_train_batches=30,  # comment in for training, running validation every 30 batches
            # fast_dev_run=True,  # check that network has no serious bugs
            callbacks=self._get_callbacks(val_loader),
            max_epochs=3 if self.config.get('fast_dev_run', False) else None,
        )

        self.trainer.fit(self.model, train_loader, val_loader)
        val_metrics = self.evaluate(val_loader)
        wandb.log(val_metrics)
        return val_metrics
        # if self.config.get('save_model', True):
        #     self.trainer.save_checkpoint(os.path.join(wandb.run.dir, 'model.ckpt'))
        # best_model_path = self.trainer.checkpoint_callback.best_model_path
        # model = self.trainer.model.load_from_checkpoint(best_model_path)
        # dst = os.path.join(wandb.run.dir, 'model.ckpt')
        # shutil.copy(best_model_path, dst)


class PIDITSTrainer(DeepARTrainer):
    def __init__(self, config: dict, logger=None):
        super().__init__(config, logger)

    def build_model(self, training):
        self.model = PIDITS.from_dataset(
            training,
            learning_rate=self.config['lr'],
            hidden_size=self.config['hidden_size'],
            dropout=self.config['dropout'],
            loss=NormalDistributionLoss(),
            log_val_interval=3,
            # reduce_on_plateau_patience=4,
            logging_metrics=self._get_logging_metrics(),
            config=self.config,
        )
        print(f"Number of parameters in the network: {self.model.size()/1e3:.1f}")


class PolarDenseTrainer(DeepARTrainer):
    def __init__(self, config: dict, logger=None):
        super().__init__(config, logger)

    def build_model(self, training):
        self.model = MyDecoderMLP.from_dataset(
            training,
            config=self.config,
            out_activation=self.config.get('out_activation', None),
            drop_input=self.config.get('drop_input', None),
            hidden_size=self.config.get('hidden_size', None),
            n_hidden_layers=self.config.get('n_hidden_layers', None),
            norm=self.config.get('norm', None),
            activation_class=self.config.get('activation_class', 'ReLU'),
            loss=RMSE(),
        )
        print(f"Number of parameters in the network: {self.model.size()/1e3:.1f}")


class PolarRNNTrainer(DeepARTrainer):
    def __init__(self, config: dict, logger=None):
        super().__init__(config, logger)

    def build_model(self, training):
        self.model = PolarRNN.from_dataset(
            training,
            learning_rate=self.config['lr'],
            hidden_size=self.config['hidden_size'],
            dropout=self.config['dropout'],
            loss=RMSE(),
            logging_metrics=self._get_logging_metrics(),
            config=self.config,
        )
        print(f"Number of parameters in the network: {self.model.size()/1e3:.1f}")


class DecoderMLPTrainer(DeepARTrainer):
    def __init__(self, config: dict, logger=None):
        super().__init__(config, logger)

    def build_model(self, training):
        self.model = DecoderMLP.from_dataset(
            training,
            # activation_class='ReLU',
            # hidden_size=300,
            # n_hidden_layers=3,
            # dropout=0.1,
            # norm=True,  # Use normalization in the MLP
            # loss=RMSE(),
            hidden_size=self.config.get('hidden_size', None),
            n_hidden_layers=self.config.get('n_hidden_layers', None),
            dropout=self.config.get('dropout', 0),
            norm=self.config.get('norm', None),
            activation_class=self.config.get('activation_class', 'ReLU'),
            loss=RMSE(),
        )
        print(f"Number of parameters in the network: {self.model.size()/1e3:.1f}")


class RNNTrainer(DeepARTrainer):
    def __init__(self, config: dict, logger=None):
        super().__init__(config, logger)

    def build_model(self, training):
        self.model = RecurrentNetwork.from_dataset(
            training,
            cell_type=self.config['cell_type'],
            hidden_size=self.config['hidden_size'],
            rnn_layers=self.config['rnn_layers'],
            dropout=self.config['dropout'],
            logging_metrics=self._get_logging_metrics(),
            loss=RMSE(),
        )
        print(f"Number of parameters in the network: {self.model.size()/1e3:.1f}")
