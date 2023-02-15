import os
from .base_mopo import mopo_params, deepcopy

params = deepcopy(mopo_params)
params.update({
    'domain': 'HalfCheetah',
    'task': 'v2',
    'exp_name': 'halfcheetah_mixed_rt_1',
    'seed': 4321,
})
params['kwargs'].update({
    'pool_load_path': os.path.expanduser('~/rds/rds-dsk-lab-eWkDxBhxBrQ/dimorl/code/dogo_results/data/MIXED-RT-1.npy'),
    # 'model_load_dir': os.path.expanduser('~/rds/rds-dsk-lab-eWkDxBhxBrQ/dimorl/code/dogo_results/mopo/ray_mopo/HalfCheetah/halfcheetah_mixed_1_101e3/seed:4321_2022-06-18_10-53-164l945x_3/models'),
    #'model_load_dir': os.path.expanduser('~/rds/rds-dsk-lab-eWkDxBhxBrQ/dimorl/code/mopo/ray_mopo/HalfCheetah/halfcheetah_mixed_rt_49_101e3/seed:1449_2023-02-13_19-12-16vz0nw7nv/models'),
    # 'model_load_dir': os.path.expanduser('~/rds/rds-dsk-lab-eWkDxBhxBrQ/dimorl/code/mopo/ray_mopo/HalfCheetah/halfcheetah_mixed_rt_49_101e3/1/models'),
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
    'bnn_batch_size': 256
})
