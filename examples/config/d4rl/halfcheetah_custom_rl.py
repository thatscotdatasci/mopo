import os
from .halfcheetah_custom import mopo_params, deepcopy

params = deepcopy(params)
params['kwargs'].update({
    'train_bnn_only': False,
    'model_load_dir': None,
})
