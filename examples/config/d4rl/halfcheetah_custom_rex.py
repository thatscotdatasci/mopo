from .halfcheetah_custom import params, deepcopy

params = deepcopy(params)
params['kwargs'].update({
    'rex': True,
    'rex_beta': 10.0,
})
