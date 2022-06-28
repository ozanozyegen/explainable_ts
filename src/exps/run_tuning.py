import os
from train.tuner import Tuner
from configs.defaults import dataset_defaults, model_defaults
os.environ['CUDA_VISIBLE_DEVICES'] = str(1)  # Select a GPU

model_name, dataset_name = 'rnn', 'synthetic'

config = dict(MODEL=model_name, DATASET=dataset_name)
config.update(dataset_defaults[dataset_name])
config.update(model_defaults[model_name])

config['is_tuning'] = False  # Uses validation dataset when True
# config['fast_dev_run'] = True

tuner = Tuner(config, tags=['tuning', 'v3'])
tuner.optimize(n_trials=100)
# best_trial = tuner.optimize(n_trials=100)
# print(f'best param: {best_trial.params}')
# print(f'best value: {best_trial.values}')