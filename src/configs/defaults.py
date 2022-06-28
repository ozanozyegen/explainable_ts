from configs.datasets import rossmann_defaults, walmart_defaults,\
    electricity_defaults, favorita_defaults, synthetic_defaults
from configs.models import mlp_defaults, gbr_defaults, emb_net_defaults,\
    polar_dense_defaults, polar_rnn_defaults, rnn_defaults


class Globs:
    project_name = 'explainable_ts'
    entity_name = ''


dataset_defaults = dict(
    rossmann=rossmann_defaults,
    walmart=walmart_defaults,
    synthetic=synthetic_defaults,
)

model_defaults = dict(
    mlp=mlp_defaults,
    rnn=rnn_defaults,
    polar_dense=polar_dense_defaults,
    polar_rnn=polar_rnn_defaults,
)
