from .hopper_custom import params, deepcopy

params = deepcopy(params)
params['kwargs'].update({
    'rex': True,
})
