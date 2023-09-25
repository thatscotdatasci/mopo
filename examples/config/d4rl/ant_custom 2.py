import os
from .base_mopo import mopo_params, deepcopy

params = deepcopy(mopo_params)
params.update({
    'domain': 'Ant',
    'task': 'v2',
    'exp_name': 'ant',
    'seed': 1234,
})
params['kwargs'].update({
    'pool_load_path': os.path.expanduser('~/rds/rds-dsk-lab-eWkDxBhxBrQ/dimorl/code/dogo_results/data/MIXED-RT-2.npy'),
    'bnn_retrain_epochs': 0,
    'pool_load_max_size': 101000,
    'rollout_length': 5,
    'rollout_batch_size': 50e3,
    'penalty_coeff': 1.0,
    'holdout_policy': None,
    'train_bnn_only': True,
    'rex': False,
    'rex_beta': 10.0,
    'rex_multiply': True,
    'repeat_dynamics_epochs': 1,
    'lr_decay': 1.0,
    'bnn_batch_size': 256,
    'hidden_dim': 1024,
    'bnn_lr': 0.0001,  # 0.001
    # 'improvement_threshold': 0.0001,
    'break_train_rex': False,
})
