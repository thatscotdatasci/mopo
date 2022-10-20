import os
import re
import json
import itertools
import collections
from glob import glob


###########################################################################################
# Update/recreate the `results_map.json` file in `dogo_results/mopo` with the latest map of
# individual experiment identifiers to their locations.
###########################################################################################


REGEX_PATTERN = r".*[A-Z][A-Z][0-9][0-9][0-9].txt"
LOCAL_RESULTS_DIR = os.path.expanduser("~/rds/hpc-work/mopo/ray_mopo")
REMOTE_RESULTS_DIR = os.path.expanduser("~/rds/rds-dsk-lab-eWkDxBhxBrQ/dimorl/code/dogo_results/mopo/ray_mopo")
RESULTS_MAP_PATH = os.path.expanduser("~/rds/rds-dsk-lab-eWkDxBhxBrQ/dimorl/code/dogo_results/mopo/results_map.json")

def glob_re(paths):
    # Identify those paths which match the pattern for experiment identifier files.
    return filter(re.compile(REGEX_PATTERN).match, paths)

def get_exp(path):
    # Extract the experiment identifier from the path of an experiment identifier file.
    return path.split('/')[-1].replace('.txt', '')

def process_path(path):
    # Extract the components of the path that will be stored in the results map.
    path_elms = path.split('/')
    return {
        "base_dir": path_elms[-3],
        "experiment_dir": path_elms[-2],
        "environment": path_elms[-4],
    }

def main():
    # Identify experiments in both the local `ray_mopo` directory, and in `dogo_results/mopo/ray_mopo`. 
    # The results are created in the local directory, before being moved to `dogo_results`.
    local_files = glob_re(glob(f"{LOCAL_RESULTS_DIR}/*/*/*/*.txt"))
    remote_files = glob_re(glob(f"{REMOTE_RESULTS_DIR}/*/*/*/*.txt"))
    all_files = itertools.chain(local_files, remote_files)

    # Create the results map by processing each of the identified paths
    results_map = {}
    for path in all_files:
        exp = get_exp(path)
        if exp in results_map:
            raise RuntimeError(f'Duplicate records for experiment: {exp}\n\n{results_map[exp]}\n\n{process_path(path)}')
        results_map[exp] = process_path(path)
    results_map = collections.OrderedDict(sorted(results_map.items()))

    # Save the results map
    with open(RESULTS_MAP_PATH, 'w') as f:
        json.dump(results_map, f, indent=4)

if __name__ == "__main__":
    main()
