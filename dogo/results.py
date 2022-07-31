import os
import re
import json
from glob import glob
from collections import namedtuple

import numpy as np
import pandas as pd
from scipy.io import loadmat

from dogo.pca.project import project_arr
from dogo.constants import (
    MOPO_BASEDIR, RESULTS_BASEDIR, SCORING_BASEDIR, MOPO_RESULTS_MAP_PATH, DYNAMICS_TRAINING_FILES, SAC_TRAINING_FILES, DEFAULT_SEED,
    STATE_DIMS, ACTION_DIMS
)

########
# Tuples
########
experiment_details = 'name base_dir experiment_dir results_dir environment dataset params seed rex rex_beta lr_decay holdout_policy elites penalty_coeff rollout_length rollout_batch_size dynamics_model_exp max_logvars min_logvars max_penalty min_penalty'.split()
ExperimentDetails = namedtuple('ExperimentDetails', experiment_details)
ExperimentResults = namedtuple('ExperimentResults', [*experiment_details, 'dynamics', 'sac'])
DynamicsTrainingResults = namedtuple('DynamicsTrainingResults', DYNAMICS_TRAINING_FILES.keys())
SacTrainingResults = namedtuple('DynamicsTrainingResults', SAC_TRAINING_FILES.keys())
PoolArrs = namedtuple('PoolArrs', 'pool pca_sa_1d pca_sa_2d mse_results explained_var_2d')

#########
# Helpers
#########
def get_results_path(experiment: str):
    with open(MOPO_RESULTS_MAP_PATH, 'r') as f:
        mopo_results_map = json.load(f)

    environment = mopo_results_map[experiment]["environment"]
    base_dir = mopo_results_map[experiment]["base_dir"]
    experiment_dir = mopo_results_map[experiment]["experiment_dir"]

    relative_path = os.path.join("ray_mopo", environment, base_dir, experiment_dir)
    local_path = os.path.join(MOPO_BASEDIR, relative_path)
    results_repo_path = os.path.join(RESULTS_BASEDIR, relative_path)
    if os.path.isdir(results_repo_path):
        return results_repo_path, environment, base_dir, experiment_dir
    elif os.path.isdir(local_path):
        return local_path, environment, base_dir, experiment_dir
    else:
        raise FileNotFoundError(f'Could not find results directory at {results_repo_path} or {local_path}')

def get_experiment_details(experiment: str, get_elites: bool = False):    
    results_dir, environment, base_dir, experiment_dir = get_results_path(experiment)

    with open(os.path.join(results_dir, "params.json"), 'r') as f:
        params = json.load(f)

    dynamics_model_exp = params["algorithm_params"]["kwargs"].get("dynamics_model_exp", None)
    if dynamics_model_exp is not None:
        dynamics_model_exp_details = get_experiment_details(dynamics_model_exp, get_elites=get_elites)

    model_weights_path = os.path.join(results_dir, 'models', "BNN_0.mat")
    if os.path.isfile(model_weights_path):
        params_dict = loadmat(model_weights_path)
        max_logvars = params_dict['14']
        min_logvars = params_dict['15']
        max_penalty = np.linalg.norm(np.sqrt(np.exp(max_logvars)))
        min_penalty = np.linalg.norm(np.sqrt(np.exp(min_logvars)))
    elif dynamics_model_exp:
        max_logvars = dynamics_model_exp_details.max_logvars
        min_logvars = dynamics_model_exp_details.min_logvars
        max_penalty = dynamics_model_exp_details.max_penalty
        min_penalty = dynamics_model_exp_details.min_penalty
    else:
        max_logvars = None
        min_logvars = None
        max_penalty = None
        min_penalty = None

    if dynamics_model_exp is not None:
        elites = dynamics_model_exp_details.elites
        rex = dynamics_model_exp_details.rex
        rex_beta = dynamics_model_exp_details.rex_beta
        lr_decay = dynamics_model_exp_details.lr_decay
        holdout_policy = dynamics_model_exp_details.holdout_policy
    else:
        if get_elites:
            with open(glob(os.path.join(results_dir, "train-log.*"))[0], 'r') as f:
                elites = json.loads(re.findall("(?<=Using 5 / 7 models: ).*", f.read())[0])
        else:
            elites = None
        rex = params["algorithm_params"]["kwargs"].get("rex", False)
        rex_beta = params["algorithm_params"]["kwargs"].get("rex_beta", None)
        lr_decay = params["algorithm_params"]["kwargs"].get("lr_decay", None)
        holdout_policy = params["algorithm_params"]["kwargs"].get("holdout_policy", None)

    return ExperimentDetails(
        name = experiment,
        base_dir = base_dir,
        experiment_dir = experiment_dir,
        results_dir = results_dir,
        environment = environment,
        dataset = params["algorithm_params"]["kwargs"]["pool_load_path"].split("/")[-1].replace('.npy', ''),
        params=params,
        seed = params["run_params"]["seed"],
        rex = rex,
        rex_beta = rex_beta,
        lr_decay = lr_decay,
        holdout_policy = holdout_policy,
        elites = elites,
        penalty_coeff = params["algorithm_params"]["kwargs"].get("penalty_coeff", None),
        rollout_length = params["algorithm_params"]["kwargs"].get("rollout_length", None),
        rollout_batch_size = params["algorithm_params"]["kwargs"].get("rollout_batch_size", None),
        dynamics_model_exp = dynamics_model_exp,
        max_logvars = max_logvars,
        min_logvars = min_logvars,
        max_penalty = max_penalty,
        min_penalty = min_penalty
    )

def get_results(experiment: str):
    experiment_details = get_experiment_details(experiment)
    
    dynamics_training_results = {}
    for k, v in DYNAMICS_TRAINING_FILES.items():
        file_path = os.path.join(experiment_details.results_dir, v)
        if os.path.isfile(file_path):
            dynamics_training_results[k] = pd.read_csv(file_path, header=None)
        else:
            dynamics_training_results[k] = None
    dynamics_training_results = DynamicsTrainingResults(**dynamics_training_results)

    sac_training_results = {}
    for k, v in SAC_TRAINING_FILES.items():
        file_path = os.path.join(experiment_details.results_dir, v)
        if os.path.isfile(file_path):
            sac_training_results[k] = pd.read_json(file_path, lines=True)
        else:
            sac_training_results[k] = None
    sac_training_results = SacTrainingResults(**sac_training_results)

    return ExperimentResults(
        **experiment_details._asdict(),
        dynamics = dynamics_training_results,
        sac = sac_training_results
    )

def get_experiment_dataset_score(experiment, dataset, seed=DEFAULT_SEED):
    with open(MOPO_RESULTS_MAP_PATH, 'r') as f:
        mopo_results_map = json.load(f)

    with open(os.path.join(SCORING_BASEDIR, mopo_results_map[experiment]['base_dir'], mopo_results_map[experiment]['experiment_dir'], f'{dataset}_{seed}.json')) as f:
        return json.load(f)

def get_pred_means_and_vars(experiment, dataset, seed=DEFAULT_SEED):
    with open(MOPO_RESULTS_MAP_PATH, 'r') as f:
        mopo_results_map = json.load(f)

    means_path = os.path.join(SCORING_BASEDIR, mopo_results_map[experiment]['base_dir'], mopo_results_map[experiment]['experiment_dir'], f'{dataset}_{seed}_means.npy')
    if os.path.isfile(means_path):
        with open(means_path, 'rb') as f:
            means = np.load(f)
    else:
        means = None
    
    vars_path = os.path.join(SCORING_BASEDIR, mopo_results_map[experiment]['base_dir'], mopo_results_map[experiment]['experiment_dir'], f'{dataset}_{seed}_vars.npy')
    if os.path.isfile(vars_path):
        with open(vars_path, 'rb') as f:
            vars = np.load(f)
    else:
        vars = None

    return means, vars

def get_scores_df(experiments: list, datasets: list):
    scores = {}
    for exp in experiments:
        exp_details = get_experiment_details(exp)
        for dataset in datasets:
            score = get_experiment_dataset_score(exp, dataset)
            score['rex'] = exp_details.rex
            score['rex_beta'] = exp_details.rex_beta or 0.
            score['seed'] = exp_details.seed
            score['training_dataset'] = exp_details.dataset
            scores[(exp_details.name, dataset)] = score
    return (
        pd.DataFrame().from_dict(scores, orient='index')[['training_dataset', 'rex', 'rex_beta', 'seed', 'overall_mse', 'observation_mse', 'reward_mse', 'log_prob', 'next_obs_log_prob', 'reward_log_prob']].
        reset_index().rename(columns={'level_0': 'experiment', 'level_1': 'evaluation_dataset'})
    )

def average_scores_over_seeds(exps: list):
    metrics = ['model_pol_total_loss_history', 'model_pol_var_loss_history', 'model_train_decay_loss_history', 'model_train_var_lim_loss_history']
    results = {m: 0. for m in metrics}
    results['model_pol_std_loss_history'] = 0.
    for exp in exps:
        for metric in metrics:
            if metric == 'model_pol_total_loss_history':
                results[metric] += (1/len(exps)) * (getattr(exp.dynamics, metric).mean(axis=1)[-50:].mean()/getattr(exp.dynamics, 'model_mean_pol_loss_history').shape[1])
            elif metric == 'model_pol_var_loss_history':
                results[metric] += (1/len(exps)) * (getattr(exp.dynamics, metric).mean(axis=1)[-50:].mean())
                results['model_pol_std_loss_history'] += (1/len(exps)) * (np.sqrt(getattr(exp.dynamics, metric)).mean(axis=1)[-50:].mean())
            else:
                results[metric] += (1/len(exps)) * (getattr(exp.dynamics, metric)[-50:].values.mean())
    return results

def get_sac_pools(exp_name, pool=None, subsample_size=10000):
    exp_details = get_experiment_details(exp_name)
    models_dir = os.path.join(exp_details.results_dir, 'models')
    
    if pool is not None:
        orig_results = np.load(os.path.join(models_dir, f'model_pool_{pool}.npy'))
        if os.path.isfile(os.path.join(models_dir, f'mse_{pool}.npy')):
            orig_mse_results = np.load(os.path.join(models_dir, f'mse_{pool}.npy'))
        else:
            orig_mse_results = None
    else:
        model_pool_files = sorted(list(glob(os.path.join(models_dir, 'model_pool_*.npy'))))
        orig_results = np.vstack([
            np.load(i) for i in model_pool_files if 'pca' not in i
        ])

        if os.path.isfile(os.path.join(models_dir, f'mse_{os.path.basename(model_pool_files[0]).split("_")[-1]}')):
            model_mse_files = [
                os.path.join(models_dir, f'mse_{os.path.basename(i).split("_")[-1]}') for i in model_pool_files
            ]
            orig_mse_results = np.vstack([
                np.load(i) for i in model_mse_files
            ])
            orig_mse_results[np.isposinf(orig_mse_results)] = np.NaN
        else:
            orig_mse_results = None

    # Subsample the results
    subsample_idxs = np.random.choice(np.arange(orig_results.shape[0]), subsample_size, replace=False)
    results =  orig_results[subsample_idxs,:]
    mse_results = orig_mse_results[subsample_idxs,:] if orig_mse_results is not None else None
    
    #######################
    # Load pre-existing PCA
    #######################
    # pca_1d_sa = np.vstack([
    #     np.load(i) for i in sorted(list(glob(os.path.join(models_dir, 'model_pool_*_pca_1d.npy'))))
    # ])[subsample_idxs,:]
    # pca_2d_sa = np.vstack([
    #     np.load(i) for i in sorted(list(glob(os.path.join(models_dir, 'model_pool_*_pca_2d.npy'))))
    # ])[subsample_idxs,:]

    pca_1d_sa, pca_2d_sa, explained_var_2d = project_arr(results[:,:STATE_DIMS+ACTION_DIMS])

    return results, pca_1d_sa, pca_2d_sa, mse_results, explained_var_2d
