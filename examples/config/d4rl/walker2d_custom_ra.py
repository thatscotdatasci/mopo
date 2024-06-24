from .walker2d_custom import params, deepcopy

params = deepcopy(params)
params['kwargs'].update({
    'rex_type': 'running_mean',
    'rex': True,
})
