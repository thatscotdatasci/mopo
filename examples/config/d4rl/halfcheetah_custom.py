from .base_mopo import mopo_params, deepcopy

params = deepcopy(mopo_params)
params.update({
    'domain': 'HalfCheetah',
    'task': 'v2',
    'exp_name': 'halfcheetah_mixed_capabilities'
})
params['kwargs'].update({
    'pool_load_path': '/home/ajc348/rds/hpc-work/mopo/rollouts/softlearning/HalfCheetah/v2/2022-05-17T18-38-36-half_cheetah_v2_3M/id=31acc_00000-seed=9479/combined_transitions.npy',
    'pool_load_max_size': 101000,
    'rollout_length': 5,
    'penalty_coeff': 1.0
})
