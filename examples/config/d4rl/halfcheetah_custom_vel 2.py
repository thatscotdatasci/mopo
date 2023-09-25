from .halfcheetah_custom import params, deepcopy

params = deepcopy(params)
params['kwargs'].update({
    'obs_indices': [4, 5, 6, 7, 13, 14, 15, 16],
})
