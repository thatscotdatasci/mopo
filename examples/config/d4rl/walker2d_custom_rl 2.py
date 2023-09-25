import os
from .walker2d_custom import params, deepcopy

params = deepcopy(params)
params['kwargs'].update({
    'train_bnn_only': False,
    'model_load_dir': None,
})


