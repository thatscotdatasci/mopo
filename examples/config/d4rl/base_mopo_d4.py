from copy import deepcopy
from .base import base_params

mopo_params = deepcopy(base_params)
mopo_params['kwargs'].update({
    'separate_mean_var': True,
    'penalty_learned_var': True,
    'bnn_retrain_epochs': 0,
    'holdout_policy': None,
    'train_bnn_only': False,
    'rex': False,
    'rex_beta': 0.,
    'rex_multiply': True,
    'repeat_dynamics_epochs': 1,
    'lr_decay': 1.0,
    'bnn_batch_size': 256,
    'rex_type': 'mopo',
    'break_train_rex': True,
})