from .halfcheetah_mixed import mopo_params, deepcopy

params = deepcopy(mopo_params)
params['kwargs'].update({
    'break_train_rex': True,
    'rex': True,
    'policy_type': 'reward_partioned',
})

