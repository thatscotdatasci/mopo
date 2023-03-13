from .walker2d_custom import params, deepcopy

params = deepcopy(params)
params['kwargs'].update({
    'rex': True,
    'rex_type': 'std',
})
