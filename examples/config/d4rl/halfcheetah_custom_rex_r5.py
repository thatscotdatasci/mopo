from .halfcheetah_custom import params, deepcopy

params = deepcopy(params)
params['kwargs'].update({
    'rex': True,
    'policy_type': 'random_5',
})
