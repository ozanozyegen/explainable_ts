mlp_defaults = {
    'lr': 0.03,
    'hidden_size': 300,
    'n_hidden_layers': 3,
    'dropout': 0.1,
    'norm': True,
    'batch_size': 128,
}

encoder_defaults = {
    'hidden_size': 300,
    'n_hidden_layers': 3,
    'dropout': 0.1,
    'norm': True,
}

emb_net_defaults = {
    'lr': 0.03,
    'batch_size': 128,
    **encoder_defaults,
}

polar_dense_defaults = {
    'lr': 0.03,
    'batch_size': 128,
    # out_activation can be changed to sigmoid for different tasks
    'out_activation': 'Identity',
    'drop_input': 0,
    'distribution_projector_bias': False,
    **encoder_defaults
}

polar_rnn_defaults = {
    'lr': 0.03,
    'hidden_size': 32,
    'dropout': 0.1,
    'cell_type': 'LSTM',
    'batch_size': 128,
    'distribution_projector_bias': False,
    'beta_rnn_bias': False,
    'alpha_rnn_bias': False,
}

rnn_defaults = {
    'cell_type': 'LSTM',
    'hidden_size': 32,
    'rnn_layers': 3,
    'dropout': 0.1,
    'batch_size': 128,
}
