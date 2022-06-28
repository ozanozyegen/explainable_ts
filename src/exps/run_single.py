from configs.defaults import dataset_defaults, model_defaults
from train.train_models import train_deep_model

model_name, dataset_name = 'polar_dense', 'synthetic'

config = dict(MODEL=model_name, DATASET=dataset_name)
config.update(dataset_defaults[dataset_name])
config.update(model_defaults[model_name])
# config['fast_dev_run'] = True

train_deep_model(config, tags=['test'])
