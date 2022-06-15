from .base_mopo import mopo_params, deepcopy

params = deepcopy(mopo_params)
params.update({
    'domain': 'HalfCheetah',
    'task': 'v2',
    'exp_name': 'halfcheetah_d3rlpy_mp1',
    'seed': 4321,
})
params['kwargs'].update({
    'pool_load_path': '/home/ajc348/rds/hpc-work/dogo_results/data/D3RLPY-MP1-P1-4.npy',
    # 'model_load_dir': '/home/ajc348/rds/hpc-work/dogo_results/mopo/ray_mopo/HalfCheetah/halfcheetah_d3rlpy_pep4_101e3/seed:1443_2022-06-01_17-07-30wh6i6w19/models',
    'pool_load_max_size': 101000,
    'rollout_length': 5,
    'penalty_coeff': 1.0,
    'holdout_policy': None,
    'train_bnn_only': False,
    'rex': False,
    'rex_beta': 10.0,
    'repeat_epochs': 1,
    'lr_decay': 1.0,
})
