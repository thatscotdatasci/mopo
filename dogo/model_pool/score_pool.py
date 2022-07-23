import argparse
import json
import os
from glob import glob

import numpy as np
from mujoco_py import MujocoException

from softlearning.environments.utils import get_environment_from_params
from dogo.results import get_experiment_details
from dogo.rollouts.split import split_halfcheetah_v2_trans_arr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy-experiment',
                        type=str,
                        help='Experiment whose dynamics model should be used.')

    args = parser.parse_args()

    return args

def get_qpos_qvel(obs):
    # Return qpos and qvel values from the current obs
    # These are necessary if setting the environment
    qpos = np.hstack((np.zeros(1),obs[:8]))
    qvel = obs[8:]
    return qpos, qvel

def score_pool(policy_experiment):
    print(f'Scoring experiment: {policy_experiment}')

    ##################################
    # Create an evaluation environment
    ##################################
    policy_exp_details = get_experiment_details(policy_experiment, get_elites=True)

    policy_exp_params_path = os.path.join(policy_exp_details.results_dir, 'params.json')
    with open(policy_exp_params_path, 'r') as f:
        policy_exp_params = json.load(f)

    environment_params = (
        policy_exp_params['environment_params']['evaluation']
        if 'evaluation' in policy_exp_params['environment_params']
        else policy_exp_params['environment_params']['training']
    )

    ##################################
    # Create an evaluation environment
    ##################################
    eval_env = get_environment_from_params(environment_params)

    ##################
    # Load Model Pools
    ##################
    model_pool_paths = glob(os.path.join(policy_exp_details.results_dir, 'models', 'model_pool_*.npy'))

    ###################
    # Score Model Pools
    ###################

    penalty_coeff = policy_exp_params['algorithm_params']['kwargs']['penalty_coeff']
    
    for model_pool_path in model_pool_paths:
        print(f'Processing: {model_pool_path}')
        mujoco_exception_count = 0

        model_pool = np.load(model_pool_path)
        obs, act, _, rew, _, _, pen = split_halfcheetah_v2_trans_arr(model_pool)
        rew_mse = np.zeros_like(rew)
        for i in range(len(model_pool)):
            pen_rew_val = rew[i,:]
            pen_val = pen[i,:]
            unpen_rew_val = pen_rew_val + penalty_coeff*pen_val
            try:
                eval_env._env.set_state(*get_qpos_qvel(obs[i,:].flatten()))
                _, rew_real, _, _ = eval_env.step(act[i,:])
            except MujocoException:
                rew_mse[i,:] = np.inf
                mujoco_exception_count += 1
            else:
                rew_mse_val = (rew_real-unpen_rew_val)**2
                rew_mse[i,:] = rew_mse_val
        print(f'Encountered {mujoco_exception_count} MujocoExceptions')
        np.save(os.path.join(policy_exp_details.results_dir, 'models', f'mse_{os.path.basename(model_pool_path).split("_")[-1]}'), rew_mse)

if __name__ == '__main__':
    args = parse_args()
    policy_experiment = args.policy_experiment
    score_pool(policy_experiment)
