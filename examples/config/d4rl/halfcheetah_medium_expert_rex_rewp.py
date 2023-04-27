from .halfcheetah_medium_expert import params, deepcopy

params = deepcopy(params)
params['kwargs'].update({
    'break_train_rex': True,
    'rex': True,
    'policy_type': 'reward_partioned',
})

