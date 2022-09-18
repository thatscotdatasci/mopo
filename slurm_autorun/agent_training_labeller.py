import os
import json
from glob import glob

def filter_function(item, params):
    return all([
        item['dynamics_model_exp'] == params['algorithm_params']['kwargs']['dynamics_model_exp'],
        item['mopo_penalty_coeff'] == params['algorithm_params']['kwargs']['penalty_coeff'],
        item['rollout_length'] == params['algorithm_params']['kwargs']['rollout_length'],
        item.get('bnn_retrain_epochs',0) == params['algorithm_params']['kwargs']['bnn_retrain_epochs']
    ])

#############
# Config File
#############
config_filepath = '/home/ajc348/rds/hpc-work/mopo/slurm_autorun/exp_params/exp_params.170922_2.json'
with open(config_filepath, 'r') as f:
    config = json.load(f)

##############
# Params Files
##############
glob_pattern = '/home/ajc348/rds/hpc-work/mopo/ray_mopo/HalfCheetah/halfcheetah_mixed_rt_5_101e3/*/params.json'
params_filepaths = glob(glob_pattern)

for params_filepath in params_filepaths:
    with open (params_filepath, 'r') as f:
        params = json.load(f)
    config_record = list(filter(lambda x: filter_function(x, params), config))
    if len(config_record) > 1:
        print(config_record)
        raise RuntimeError(params)
    elif len(config_record) == 0:
        continue
    
    with open(os.path.join(os.path.dirname(params_filepath), f'{config_record[0]["exp_id"]}.txt'), 'w') as f:
        f.write('')

    print(config_record[0])
