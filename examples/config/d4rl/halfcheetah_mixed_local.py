from .base_mopo import mopo_params, deepcopy

params = deepcopy(mopo_params)
params.update({
    'domain': 'HalfCheetah',
    'task': 'v2',
    'exp_name': 'halfcheetah_medium_replay_local'
})
params['kwargs'].update({
    'pool_load_path': '/home/ajc348/rds/hpc-work/mopo/d_4rl_halfcheetah_medium_replay_v0.npy',
    'pool_load_max_size': 101000,
    'rollout_length': 5,
    'penalty_coeff': 1.0
})
