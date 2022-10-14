import os
import re
import json
import itertools
import collections
from glob import glob

REGEX_PATTERN = r".*[A-Z][A-Z][0-9][0-9][0-9].txt"
LOCAL_RESULTS_DIR = "/home/ajc348/rds/hpc-work/mopo/ray_mopo"
REMOTE_RESULTS_DIR = "/home/ajc348/rds/rds-dsk-lab-eWkDxBhxBrQ/dimorl/code/dogo_results/mopo/ray_mopo"
RESULTS_MAP_PATH = "/home/ajc348/rds/rds-dsk-lab-eWkDxBhxBrQ/dimorl/code/dogo_results/mopo/results_map.json"

def glob_re(paths):
    return filter(re.compile(REGEX_PATTERN).match, paths)

def get_exp(path):
    return path.split('/')[-1].replace('.txt', '')

def process_path(path):
    path_elms = path.split('/')
    return {
        "base_dir": path_elms[-3],
        "experiment_dir": path_elms[-2],
        "environment": path_elms[-4],
    }

def main():
    local_files = glob_re(glob(f"{LOCAL_RESULTS_DIR}/*/*/*/*.txt"))
    remote_files = glob_re(glob(f"{REMOTE_RESULTS_DIR}/*/*/*/*.txt"))
    all_files = itertools.chain(local_files, remote_files)

    results_map = {}
    for path in all_files:
        exp = get_exp(path)
        if exp in results_map:
            raise RuntimeError(f'Duplicate records for experiment: {exp}\n\n{results_map[exp]}\n\n{process_path(path)}')
        results_map[exp] = process_path(path)
    results_map = collections.OrderedDict(sorted(results_map.items()))

    with open(RESULTS_MAP_PATH, 'w') as f:
        json.dump(results_map, f, indent=4)

if __name__ == "__main__":
    main()
