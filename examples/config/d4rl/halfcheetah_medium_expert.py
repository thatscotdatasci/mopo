from .base_mopo import mopo_params, deepcopy

params = deepcopy(mopo_params)
params.update({
    'domain': 'halfcheetah',
    'task': 'medium-expert-v0',
    'exp_name': 'halfcheetah_medium_expert',
    'seed': 1443,
})
params['kwargs'].update({
    'pool_load_path': 'd4rl/halfcheetah-medium-expert-v0',
    'pool_load_max_size': 2 * 10**6,
    'rollout_length': 5,
    'penalty_coeff': 5.0,
    'holdout_policy': None,
    'train_bnn_only': False,
    'rex': False,
    'rex_beta': 10.0,
    'rex_multiply': True,
    'repeat_dynamics_epochs': 0,
    'lr_decay': 0.1,
})