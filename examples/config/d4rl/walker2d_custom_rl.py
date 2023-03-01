import os
from .walker2d_custom import params, deepcopy

params = deepcopy(params)
params['kwargs'].update({
    # 'model_load_dir':  '/home/hpcanok1/rds/rds-dsk-lab-eWkDxBhxBrQ/dimorl/code/mopo/ray_mopo/Walker2d_W-MIXED-RT-1/default/s12_2023-02-27_15-47-18bs6_l049/models',
    'train_bnn_only': False,
    'model_load_dir': None,
})


