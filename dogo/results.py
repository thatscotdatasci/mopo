import os
import json
from collections import namedtuple

import pandas as pd

from dogo.constants import (
    RESULTS_BASEDIR, SCORING_BASEDIR, MOPO_RESULTS_MAP_PATH, DYNAMICS_TRAINING_FILES, SAC_TRAINING_FILES, DEFAULT_SEED
)

########
# Tuples
########
experiment_details = 'name base_dir experiment_dir results_dir environment dataset params seed rex rex_beta holdout_policy'.split()
ExperimentDetails = namedtuple('ExperimentDetails', experiment_details)
ExperimentResults = namedtuple('ExperimentResults', [*experiment_details, 'dynamics', 'sac'])
DynamicsTrainingResults = namedtuple('DynamicsTrainingResults', DYNAMICS_TRAINING_FILES.keys())
SacTrainingResults = namedtuple('DynamicsTrainingResults', SAC_TRAINING_FILES.keys())

#########
# Helpers
#########
def get_results_path(local_path: str):
    results_repo_path = os.path.join(RESULTS_BASEDIR, local_path)
    if os.path.isfile(results_repo_path):
        return results_repo_path
    else:
        return local_path

def get_experiment_details(experiment: str):
    with open(MOPO_RESULTS_MAP_PATH, 'r') as f:
        mopo_results_map = json.load(f)

    environment = mopo_results_map[experiment]["environment"]
    base_dir = mopo_results_map[experiment]["base_dir"]
    experiment_dir = mopo_results_map[experiment]["experiment_dir"]

    results_dir = os.path.join("ray_mopo", environment, base_dir, experiment_dir)
    repo_results_dir = os.path.join(RESULTS_BASEDIR, results_dir)
    if os.path.isdir(repo_results_dir):
        results_dir = repo_results_dir

    with open(os.path.join(results_dir, "params.json"), 'r') as f:
        params = json.load(f)

    return ExperimentDetails(
        name = experiment,
        base_dir = base_dir,
        experiment_dir = experiment_dir,
        results_dir = results_dir,
        environment = environment,
        dataset = params["algorithm_params"]["kwargs"]["pool_load_path"].split("/")[-1][:-4],
        params=params,
        seed = params["run_params"]["seed"],
        rex = params["algorithm_params"]["kwargs"].get("rex", False),
        rex_beta = params["algorithm_params"]["kwargs"].get("rex_beta", None),
        holdout_policy = params["algorithm_params"]["kwargs"].get("holdout_policy", None),
    )

def get_results(experiment: str):
    experiment_details = get_experiment_details(experiment)
    
    dynamics_training_results = {}
    for k, v in DYNAMICS_TRAINING_FILES.items():
        file_path = os.path.join(experiment_details.results_dir, v)
        dynamics_training_results[k] = pd.read_csv(file_path, header=None)
    dynamics_training_results = DynamicsTrainingResults(**dynamics_training_results)

    sac_training_results = {}
    for k, v in SAC_TRAINING_FILES.items():
        file_path = os.path.join(experiment_details.results_dir, v)
        sac_training_results[k] = pd.read_json(file_path, lines=True)
    sac_training_results = SacTrainingResults(**sac_training_results)

    return ExperimentResults(
        **experiment_details._asdict(),
        dynamics = dynamics_training_results,
        sac = sac_training_results
    )

def get_score(experiment, dataset, seed=DEFAULT_SEED):
    with open(MOPO_RESULTS_MAP_PATH, 'r') as f:
        mopo_results_map = json.load(f)

    with open(os.path.join(SCORING_BASEDIR, mopo_results_map[experiment]['base_dir'], mopo_results_map[experiment]['experiment_dir'], f'{dataset}_{seed}.json')) as f:
        return json.load(f)
