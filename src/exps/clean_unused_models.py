from turtle import clear
from configs.defaults import Globs
from helpers.wandb_common import clean_removed_model_files

clean_removed_model_files(Globs.project_name,
                          Globs.entity_name,
                          save_dir='models/')
