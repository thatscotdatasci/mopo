import os
import json
from glob import glob

def main(results_dir: str):    
    exp_dirs = glob(os.path.join(results_dir, 'seed:*'))

    for exp_dir in exp_dirs:
        params_path = os.path.join(exp_dir, 'params.json')

        with open(params_path, 'r') as f:
            params_json = json.load(f)

        with open(params_path, 'w') as f:
            json.dump(params_json, f, indent=4)

if __name__ == "__main__":
    # results_dir = sys.argv[1]
    results_dir = os.path.abspath('ray_mopo/HalfCheetah/halfcheetah_mixed_rt_1_101e3')
    main(results_dir)
