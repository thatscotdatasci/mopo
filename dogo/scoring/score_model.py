import os
import sys
import json
from typing import List

import numpy as np
import tensorflow as tf

from mopo.models.bnn import BNN
from mopo.off_policy.loader import restore_pool_contiguous
from softlearning.environments.utils import get_environment_from_params
from softlearning.replay_pools.utils import get_replay_pool_from_variant
from dogo.results import get_experiment_details

DATA_BASEDIR = os.path.expanduser("~/rds/rds-dsk-lab-eWkDxBhxBrQ/dimorl/code/dogo_results/data")
DATA_PATHS = []

PARAMETERS_PATH_HC = os.path.expanduser("~/rds/rds-dsk-lab-eWkDxBhxBrQ/dimorl/code/mopo/dogo/bnn_params_halfcheetah.json")
PARAMETERS_PATH_H = os.path.expanduser("~/rds/rds-dsk-lab-eWkDxBhxBrQ/dimorl/code/mopo/dogo/bnn_params_hopper.json")
PARAMETERS_PATH_W = os.path.expanduser("~/rds/rds-dsk-lab-eWkDxBhxBrQ/dimorl/code/mopo/dogo/bnn_params_walker2d.json")
OUTPUT_BASE_DIR = os.path.expanduser("~/rds/rds-dsk-lab-eWkDxBhxBrQ/dimorl/code/dogo_results/mopo/model_scoring")


def score_model(experiment: str, data_paths: List[str], deterministic=True):
    """ Use the environment model trained in `experiment` to make predictions for the next state and reward for those
    transitions in the `data_paths` files. Determine the log-likelihood of the real next state and reward, and the MSE
    of the predictions.
    """
    exp_details = get_experiment_details(experiment, get_elites=True)
    model_dir = os.path.join(exp_details.results_dir, 'models')
    elites = exp_details.elites

    # Load parameters - the mininal necessary given we are loading a pre-trained model
    if exp_details.environment == 'HalfCheetah':
        parameters_path = PARAMETERS_PATH_HC
    elif exp_details.environment == 'Hopper':
        parameters_path = PARAMETERS_PATH_H
    elif exp_details.environment == 'Walker2d':
        parameters_path = PARAMETERS_PATH_HC
    else:
        raise RuntimeError(f'No parameters file for environment: {exp_details.environment}')
    
    with open(parameters_path, 'r') as f:
        params = json.load(f)
    params["model_dir"] = model_dir

    # Set the seed - use the same seed as applied in the environment model training
    seed = params['seed']
    if seed is not None:
        np.random.seed(seed)

    ##########################
    # Get training environment
    ##########################
    # Taken from: examples/development/main.py

    training_environment = get_environment_from_params(params['environment_params']['training'])

    ###################
    # Instantiate model
    ###################
    # This will load the model whose location is specified in the parameters
    # Taken from: ~/rds/rds-dsk-lab-eWkDxBhxBrQ/dimorl/code/mopo/mopo/models/constructor.py

    bnn = BNN(params)
    bnn.model_loaded = True
    bnn.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})

    for data_path in data_paths:
        data_path = os.path.join(DATA_BASEDIR, data_path)

        #####################################
        # Create and load replay pool/dataset
        #####################################
        # Taken from: examples/development/main.py

        params["replay_pool_params"]["pool_load_path"] = data_path
        replay_pool = get_replay_pool_from_variant(params, training_environment)

        # Usually called by restore_pool, which is called in mopo/algorithms/mopo.py
        restore_pool_contiguous(replay_pool, params['replay_pool_params']['pool_load_path'])
        env_samples = replay_pool.return_all_samples()
        
        #######################################################################################
        # Run obs and actions through model to obtained predicted rewards and next observations
        #######################################################################################
        # Taken from the `step` method in FakeEnv: mopo/models/fake_env.py

        obs, acts, rewards, next_obs = env_samples['observations'], env_samples['actions'], env_samples['rewards'], env_samples['next_observations']
        inputs = np.concatenate((obs, acts), axis=-1)
        outputs = np.concatenate((rewards, next_obs), axis=-1)

        # Note that the dynamics model actually predicts the difference between the current and next observation, hence we add the original obs
        # to predicted means
        ensemble_model_means, ensemble_model_vars = bnn.predict(inputs, factored=True)
        ensemble_model_means[:,:,1:] += obs
        ensemble_model_stds = np.sqrt(ensemble_model_vars)
    
        # The first dimension of the returned matrices corresponds to the predicted reward, the rest corresponds to the predicted next observation
        ensemble_reward_model_means, ensemble_next_obs_model_means = ensemble_model_means[:,:,:1], ensemble_model_means[:,:,1:]
        ensemble_reward_model_vars, ensemble_next_obs_model_vars = ensemble_model_vars[:,:,:1], ensemble_model_vars[:,:,1:]

        # We only sample from the predictive Gaussian distribution if the `deterministic` argument is True (default)
        # Otherwise the mean prediction is used
        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

        pred_rewards, pred_next_obs = ensemble_samples[:,:,:1], ensemble_samples[:,:,1:]
        pred_outputs = np.concatenate((pred_rewards, pred_next_obs), axis=-1)

        ##########################
        # Determine Log-Likelihood
        ##########################
        # Taken from the `step` method in FakeEnv: mopo/models/fake_env.py
        # Determine the combined log-probability, as well as the individual reward and dynamics log-probabilities

        k = outputs.shape[-1]
        log_probs = -1/2 * (k * np.log(2*np.pi) + np.log(ensemble_model_vars).sum(-1) + (np.power(outputs-ensemble_model_means, 2)/ensemble_model_vars).sum(-1))
        log_prob = log_probs.mean(axis=-1)

        reward_log_probs = -1/2 * (1 * np.log(2*np.pi) + np.log(ensemble_reward_model_vars).sum(-1) + (np.power(rewards-ensemble_reward_model_means, 2)/ensemble_reward_model_vars).sum(-1))
        reward_log_prob = reward_log_probs.mean(axis=-1)

        next_obs_log_probs = -1/2 * ((k-1) * np.log(2*np.pi) + np.log(ensemble_next_obs_model_vars).sum(-1) + (np.power(next_obs-ensemble_next_obs_model_means, 2)/ensemble_next_obs_model_vars).sum(-1))
        next_obs_log_prob = next_obs_log_probs.mean(axis=-1)
        
        ###############
        # Determine MSE
        ###############
        # Taking the mean over both features and records
        # Replicates the way the MSE is calculated in the `_compile_losses` method of mopo/models/bnn.py (i.e. taking the mean over features, rather than the vector norm)
        # Again, determine the combined log-probability, as well as the individual reward and dynamics log-probabilities

        actual_rewards, actual_next_obs = env_samples['rewards'], env_samples['next_observations']
        overall_mse = ((outputs-pred_outputs)**2).mean(axis=-1).mean(axis=-1)
        reward_mse = ((actual_rewards-pred_rewards)**2).mean(axis=-1).mean(axis=-1)
        obs_mse = ((actual_next_obs-pred_next_obs)**2).mean(axis=-1).mean(axis=-1)

        #############################
        # Print the results to stdout
        #############################

        print(f'Model Directory: {params["model_dir"]}')
        print(f'Data Used: {params["replay_pool_params"]["pool_load_path"]}')
        print(f'Reward MSE: {reward_mse[elites].mean()} | Observation MSE: {obs_mse[elites].mean()} | log_prob: {log_prob[elites].mean()}')

        ##############################
        # Save a JSON with the results
        ##############################

        json_results = {
            "model_dir": model_dir,
            "data_path": data_path,
            "seed": seed,
            "elites": exp_details.elites,
            "deterministic": deterministic,
            "overall_mse": float(overall_mse[elites].mean()),
            "overall_mses": overall_mse.tolist(),
            "reward_mse": float(reward_mse[elites].mean()),
            "reward_mses": reward_mse.tolist(),
            "observation_mse": float(obs_mse[elites].mean()),
            "observation_mses": obs_mse.tolist(),
            "log_prob": float(log_prob[elites].mean()),
            "log_probs": log_prob.tolist(),
            "reward_log_prob": float(reward_log_prob[elites].mean()),
            "reward_log_probs": reward_log_prob.tolist(),
            "next_obs_log_prob": float(next_obs_log_prob[elites].mean()),
            "next_obs_log_probs": next_obs_log_prob.tolist(),
        }

        json_output_dir = os.path.join(OUTPUT_BASE_DIR, model_dir.split("/")[-3], model_dir.split("/")[-2])
        if not os.path.isdir(json_output_dir):
            os.makedirs(json_output_dir)
        
        json_output_path = os.path.join(json_output_dir, f'{data_path.split("/")[-1][:-4]}_{seed}.json')
        with open(json_output_path, 'w') as f:
            json.dump(json_results, f, indent=4)

        # Save the means and variances - these are used to calculate Wasserstein distances, and analysis of the
        # standard deviations tells us about the reward penalties MOPO is likely to apply during training.
        # However, this is quite data hungry, and so could potentially be disabled.
        with open(os.path.join(json_output_dir, f'{data_path.split("/")[-1][:-4]}_{seed}_means.npy'), 'wb') as f:
            np.save(f, ensemble_model_means)
        with open(os.path.join(json_output_dir, f'{data_path.split("/")[-1][:-4]}_{seed}_vars.npy'), 'wb') as f:
            np.save(f, ensemble_model_vars)


if __name__ == "__main__":
    experiment = sys.argv[1]
    score_model(experiment, DATA_PATHS)
