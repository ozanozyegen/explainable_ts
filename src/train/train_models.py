from collections import defaultdict
import os, math
import pickle
import wandb
import numpy as np

from pytorch_lightning.loggers import TensorBoardLogger

from helpers.wandb_common import wandb_save
from configs.defaults import Globs
from data.loader import data_loader
from train.trainer import select_trainer


def train_deep_model(config: dict, tags=[]):
    wandb_save(is_online=True)
    wandb.init(project=Globs.project_name, entity=Globs.entity_name,
        config=config, tags=tags, reinit=True,
        sync_tensorboard=True)
    # wandb.define_metric('val_NRMSE', summary='min')
    # wandb.define_metric('val_ND', summary='min')

    logger = TensorBoardLogger("tensorboard_logs", name=wandb.run.id)

    dataset_formatter = data_loader(config['DATASET'])(config)
    training, train_loader, val_loader = dataset_formatter.load_dataset()

    trainer = select_trainer(config['MODEL'])(config, logger)
    trainer.build_model(training)
    val_metrics = trainer.train(train_loader, val_loader)
    if 'tuning' in tags:
        return val_metrics, config


def train_sk_models(config: dict, tags=[]):
    wandb_save(is_online=True)
    wandb.init(project=Globs.project_name, entity=Globs.entity_name,
        config=config, tags=tags, reinit=True)

    dataset_formatter = data_loader(config['DATASET'])(config)
    data_generator = dataset_formatter.load_sk_dataset()
    val_metrics_list = []
    pred_package_list = []
    for dataset in data_generator:
        trainer = select_trainer(config['MODEL'])(config, wandb)
        trainer.build_model()
        val_metrics, pred_package = trainer.train(dataset)
        val_metrics_list.append(val_metrics)
        pred_package_list.append(pred_package)
    # Average the results
    pred_y, test_y, pred_q, quants = [], [], [], pred_package_list[0][3]
    for pred_package in pred_package_list:
        pred_y.append(pred_package[0])
        test_y.append(pred_package[1])
        pred_q.append(pred_package[2])
    pred_y = np.concatenate(pred_y, axis=0)
    test_y = np.concatenate(test_y, axis=0)
    pred_q = np.concatenate(pred_q, axis=0)
    # Calculate metrics
    val_metrics = trainer.evaluate_metrics(pred_y, test_y, pred_q, quants)
    wandb.log(val_metrics)
    print(val_metrics)

    # NOTE: Save NRMSE2, ND2.. with removed 0's from metric calculations
    good_idxs = test_y > 0
    val_metrics2 = trainer.evaluate_metrics(
        pred_y[good_idxs], test_y[good_idxs], pred_q[good_idxs], quants,
        suffix='2'
    )
    wandb.log(val_metrics2)
    print(val_metrics2)

    if 'tuning' in tags:
        return val_metrics, config
