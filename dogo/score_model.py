import json

import numpy as np
import tensorflow as tf

from mopo.models.bnn import BNN
from mopo.models.constructor import format_samples_for_training
from mopo.models.fake_env import FakeEnv
from mopo.off_policy.loader import restore_pool_contiguous
from softlearning.environments.utils import get_environment_from_params
from softlearning.replay_pools.utils import get_replay_pool_from_variant

PARAMETERS_PATH = "/home/ajc348/rds/hpc-work/mopo/dogo/bnn_params.json"

def main():
    # Load parameters - the mininal needed give we are loading a pre-trained model
    with open(PARAMETERS_PATH, 'r') as f:
        params = json.load(f)

    # Set the seed
    np.random.seed(params['seed'])

    ##########################
    # Get training environment
    ##########################
    # From: examples/development/main.py

    training_environment = get_environment_from_params(params['environment_params']['training'])

    #####################################
    # Create and load replay pool/dataset
    #####################################
    # From: examples/development/main.py

    replay_pool = get_replay_pool_from_variant(params, training_environment)

    # Usually called by restore_pool, which is called in mopo/algorithms/mopo.py
    restore_pool_contiguous(replay_pool, params['replay_pool_params']['pool_load_path'])
    env_samples = replay_pool.return_all_samples()
    
    ###################
    # Instantiate model
    ###################
    # This will load the model whose location is specified in the parameters
    # From: /rds/user/ajc348/hpc-work/mopo/mopo/models/constructor.py

    bnn = BNN(params)
    bnn.model_loaded = True
    bnn.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})

    ###################################
    # Run obs and actions through model
    ###################################
    # From the `step` method in FakeEnv: mopo/models/fake_env.py

    obs, acts = env_samples['observations'], env_samples['actions']
    inputs = np.concatenate((obs, acts), axis=-1)
    ensemble_model_means, ensemble_model_vars = bnn.predict(inputs, factored=True)
    ensemble_model_means[:,:,1:] += obs
    ensemble_model_stds = np.sqrt(ensemble_model_vars)
    ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds
    
    samples = np.mean(ensemble_samples, axis=0)
    model_means = np.mean(ensemble_model_means, axis=0)
    model_stds = np.mean(ensemble_model_stds, axis=0)

    log_prob, dev = FakeEnv._get_logprob(None, samples, ensemble_model_means, ensemble_model_vars)

    pred_rewards, pred_next_obs = samples[:,:1], samples[:,1:]
    
    ###############
    # Determine MSE
    ###############
    actual_rewards, actual_next_obs = env_samples['rewards'], env_samples['next_observations']
    reward_mse = ((actual_rewards-pred_rewards)**2).mean()
    obs_mse = ((actual_next_obs-pred_next_obs)**2).mean()

    print(f'Model Directory: {params["model_dir"]}')
    print(f'Data Used: {params["replay_pool_params"]["pool_load_path"]}')
    print(f'Reward MSE: {reward_mse} | Observation MSE: {obs_mse} | Mean Log Prob: {log_prob.mean()}')

    return reward_mse, obs_mse, log_prob.mean()


if __name__ == "__main__":
    main()

