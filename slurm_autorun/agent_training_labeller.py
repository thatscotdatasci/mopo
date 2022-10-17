import os
import json
from glob import glob


##############################################################################################
# Label experiments using the exp_params file that was used to define the experiments, and run
# them on the HPC
##############################################################################################


def filter_function(item, params):
    """ This function will be passed to the `filter` function - the passed `item` will only be
    retained if it meets the below conditions.
    """
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
# This will return all params.json files in the given results directory, not just those relating to
# experiments defined in the exp_params file.
glob_pattern = '/home/ajc348/rds/hpc-work/mopo/ray_mopo/HalfCheetah/halfcheetah_mixed_rt_5_101e3/*/params.json'
params_filepaths = glob(glob_pattern)


# Loop through the parameters files, only processing those defined in the exp_params file.
for params_filepath in params_filepaths:
    with open (params_filepath, 'r') as f:
        params = json.load(f)
    config_record = list(filter(lambda x: filter_function(x, params), config))

    if len(config_record) > 1:
        # It shouldn't be possible for an experiment to match with more than one entry in the exp_params file.
        print(config_record)
        raise RuntimeError(params)
    elif len(config_record) == 0:
        # It's fine if the experiment does not appear in the exp_params file.
        # We might have triggered experiments using multiple exp_params files, or not moved old results.
        continue
    
    # Create an empty file with the correct label
    # Really we should also check for an existing label file and throw an error if one is already present
    with open(os.path.join(os.path.dirname(params_filepath), f'{config_record[0]["exp_id"]}.txt'), 'w') as f:
        f.write('')

    # Print the configuration of the experiment that has been labelled
    print(config_record[0])
