import os
import json
from glob import glob


##################################################################################
# Ray saves a JSON containing the parameters used in a given run as a single line.
# This script simply breaks this onto multiple lines and adds indentations.
##################################################################################


def main(results_dir: str):    
    # Identify all the individual experiment directories
    exp_dirs = glob(os.path.join(results_dir, 'seed:*'))

    # Loop through the directories
    for exp_dir in exp_dirs:
        # Define the parameters path
        params_path = os.path.join(exp_dir, 'params.json')

        # Load the current parameters file
        with open(params_path, 'r') as f:
            params_json = json.load(f)

        # Overwrite the current parameters file with a pretified version
        with open(params_path, 'w') as f:
            json.dump(params_json, f, indent=4)


if __name__ == "__main__":
    # results_dir = sys.argv[1]
    results_dir = os.path.abspath('ray_mopo/HalfCheetah/halfcheetah_mixed_rt_1_101e3')
    main(results_dir)
