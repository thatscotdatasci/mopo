import os
from .walker2d_custom import params, deepcopy

params = deepcopy(params)
params['kwargs'].update({
    'train_bnn_only': False,
    'model_load_dir': None,
    'hidden_dim': 1024,
    'rex_type': 'running_mean',
})


