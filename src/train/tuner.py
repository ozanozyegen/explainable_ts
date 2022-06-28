import os, json, shutil
import joblib
import optuna, wandb
from optuna.integration.wandb import WeightsAndBiasesCallback
from configs.defaults import Globs
from train.train_models import train_deep_model, train_sk_models


class Tuner(object):
    def __init__(self, config, tags=['tuning']):
        self.config = config
        self.model = config['MODEL']
        self.metric = 'final_ND'
        self._deep_models = ['mlp', 'polar_dense', 'polar_rnn', 'rnn']
        self._tags = tags

    def optimization_function(self, trial: optuna.trial.Trial):
        if self.model == 'polar_rnn':
            self.config['lr'] = trial.suggest_loguniform('lr', 1e-4, 1e-1)
            self.config['hidden_size'] = trial.suggest_int('hidden_size', 16, 128)
            self.config['dropout'] = trial.suggest_float("dropout", 0, 0.5)
            self.config['cell_type'] = trial.suggest_categorical('cell_type', ['LSTM', 'GRU'])
            self.config['rnn_layers'] = trial.suggest_int('rnn_layers', 1, 8)
        elif self.model == 'emb_net':
            raise ValueError('Not defined')
        elif self.model == 'polar_dense':
            self.config['hidden_size'] = trial.suggest_int("hidden_size", 100, 1000)
            self.config['n_hidden_layers'] = trial.suggest_int("n_hidden_layers", 1, 8)
            self.config['dropout'] = trial.suggest_float("dropout", 0, 0.5)
            self.config['norm'] = trial.suggest_categorical('norm', [True, False])
            self.config['lr'] = trial.suggest_loguniform('lr', 1e-4, 1e-1)
        elif self.model == 'mlp':
            self.config['hidden_size'] = trial.suggest_int("hidden_size", 100, 1000)
            self.config['n_hidden_layers'] = trial.suggest_int("n_hidden_layers", 1, 8)
            self.config['dropout'] = trial.suggest_float("dropout", 0, 0.5)
            self.config['norm'] = trial.suggest_categorical('norm', [True, False])
            self.config['lr'] = trial.suggest_loguniform('lr', 1e-4, 1e-1)
        elif self.model == 'rnn':
            self.config['lr'] = trial.suggest_loguniform('lr', 1e-4, 1e-1)
            self.config['hidden_size'] = trial.suggest_int('hidden_size', 16, 128)
            self.config['dropout'] = trial.suggest_float("dropout", 0, 0.5)
            self.config['cell_type'] = trial.suggest_categorical('cell_type', ['LSTM', 'GRU'])
            self.config['rnn_layers'] = trial.suggest_int('rnn_layers', 1, 8)

        print('Training with params: ', self.config)
        if self.model in self._deep_models:
            val_metrics, config_dict = train_deep_model(self.config, tags=self._tags)
        else:  # Sklearn model
            val_metrics, config_dict = train_sk_models(self.config, tags=self._tags)

        trial.set_user_attr(key='val_metrics', value=val_metrics)
        trial.set_user_attr(key='config_dict', value=config_dict)
        return val_metrics['final_ND']

    def optimize(self, direction='minimize', n_trials=100):
        study_name = f"{self.config['DATASET']}_{self.config['MODEL']}_" + '_'.join(self._tags)
        study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            storage=
            f"sqlite:///data/interim/optuna.db",
            load_if_exists=True)
        callbacks = []  # wandbc
        study.optimize(self.optimization_function, n_trials=n_trials,
                       callbacks=callbacks)