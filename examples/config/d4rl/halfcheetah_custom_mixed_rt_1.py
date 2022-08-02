from .base_mopo import mopo_params, deepcopy

params = deepcopy(mopo_params)
params.update({
    'domain': 'HalfCheetah',
    'task': 'v2',
    'exp_name': 'halfcheetah_mixed_rt_1',
    'seed': 144,
})
params['kwargs'].update({
    'pool_load_path': '/home/ajc348/rds/hpc-work/dogo_results/data/MIXED-RT-1.npy',
    # 'model_load_dir': '/home/ajc348/rds/hpc-work/dogo_results/mopo/ray_mopo/HalfCheetah/halfcheetah_mixed_3_101e3/seed:1443_2022-07-12_18-45-29jgyuaam3/models',
    'bnn_retrain_epochs': 1,
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
