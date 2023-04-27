from .halfcheetah_mixed import params, deepcopy

params = deepcopy(params)
params['kwargs'].update({
    'break_train_rex': True,
    'rex': True,
    'policy_type': 'random_5',
})

