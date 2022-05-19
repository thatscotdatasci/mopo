from .base_mopo import mopo_params, deepcopy

params = deepcopy(mopo_params)
params.update({
    'domain': 'HalfCheetah',
    'task': 'v2',
    'exp_name': 'halfcheetah_d3rlpy_pep2'
})
params['kwargs'].update({
    'pool_load_path': '/home/ajc348/rds/hpc-work/mopo/rollouts/D3RLPY-PEP2.npy',
    'pool_load_max_size': 101000,
    'rollout_length': 5,
    'penalty_coeff': 1.0
})
