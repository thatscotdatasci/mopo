import os
import re
import json
import itertools
import collections
from glob import glob

RESULTS_MAP_PATH = "/home/ajc348/rds/rds-dsk-lab-eWkDxBhxBrQ/dimorl/code/dogo_results/mopo/results_map.json"
RESULTS_MAP_ORDER_PATH = "/home/ajc348/rds/rds-dsk-lab-eWkDxBhxBrQ/dimorl/code/dogo_results/mopo/results_map_order.json"

def main():
    with open(RESULTS_MAP_PATH, 'r') as f:
        results_map = json.load(f)

    results_map = collections.OrderedDict(sorted(results_map.items()))

    with open(RESULTS_MAP_ORDER_PATH, 'w') as f:
        json.dump(results_map, f, indent=4)

if __name__ == "__main__":
    main()
