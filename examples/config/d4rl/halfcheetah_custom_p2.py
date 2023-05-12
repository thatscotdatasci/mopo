import os
from .halfcheetah_custom import params, deepcopy

params = deepcopy(params)
params['kwargs'].update({
    'obs_indices': [3, 8, 15, 16],
})
