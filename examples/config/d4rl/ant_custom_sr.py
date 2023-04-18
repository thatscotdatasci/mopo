from .ant_custom import params, deepcopy

params = deepcopy(params)
params['kwargs'].update({
    'rex_type': 'scale_reward',
})
