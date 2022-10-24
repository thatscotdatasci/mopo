import argparse
import json
import os
from glob import glob

import numpy as np
from mujoco_py import MujocoException

from softlearning.environments.utils import get_environment_from_params
from dogo.results import get_experiment_details
from dogo.rollouts.split import split_halfcheetah_v2_trans_arr, split_hopper_v2_trans_arr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy-experiment',
                        type=str,
                        help='Experiment whose dynamics model should be used.')
    parser.add_argument('--env-name',
                        type=str,
                        choices=['HalfCheetah', 'Walker2d', 'Hopper'],
                        default='HalfCheetah',
                        help='Environment')

    args = parser.parse_args()

    return args

def get_qpos_qvel(obs, env_name):
    # Return qpos and qvel values from the current obs
    # These are necessary if setting the environment
    pos_vel_split_ind = 5 if env_name == 'Hopper' else 8
    qpos = np.hstack((np.zeros(1),obs[:pos_vel_split_ind]))
    qvel = obs[pos_vel_split_ind:]
    return qpos, qvel

def score_pool(policy_experiment, env_name):
    print(f'Scoring experiment: {policy_experiment}')
    print(f'Environment: {env_name}')

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

        if env_name == 'Hopper':
            obs, act, next_obs, rew, _, _, pen = split_hopper_v2_trans_arr(model_pool)
        else:
            obs, act, next_obs, rew, _, _, pen = split_halfcheetah_v2_trans_arr(model_pool)
        
        overall_mse = np.zeros_like(rew)
        rew_mse = np.zeros_like(rew)
        for i in range(len(model_pool)):
            # Determine the value of the unpenalised reward
            pen_rew_val = rew[i,:]
            pen_val = pen[i,:]
            unpen_rew_val = pen_rew_val + penalty_coeff*pen_val

            try:
                eval_env._env.set_state(*get_qpos_qvel(obs[i,:].flatten(), env_name))
                next_obs_real, rew_real, _, _ = eval_env.step(act[i,:])
            except MujocoException:
                # If the MuJoCo environment throws an exception, record infinite errors
                # This will occur when the action is degenerate
                rew_mse[i,:] = np.inf
                overall_mse[i,:] = np.inf
                mujoco_exception_count += 1
            else:
                rew_mse_val = (rew_real-unpen_rew_val)**2
                rew_mse[i,:] = rew_mse_val

                next_obs_rew = np.hstack((next_obs[i,:], unpen_rew_val))
                next_obs_rew_real = np.hstack((next_obs_real['observations'], rew_real))

                # Rather than taking the mean MSE over the features, I take the vector norm
                # This seems a more natural approach, but does not match what is done in the MOPO code
                # Consequently, it does not match the approach in the model scoring scripts
                next_obs_rew_mse = np.linalg.norm((next_obs_rew_real-next_obs_rew)**2)
                overall_mse[i,:] = next_obs_rew_mse

        print(f'Encountered {mujoco_exception_count} MujocoExceptions')

        # Save the results
        np.save(os.path.join(policy_exp_details.results_dir, 'models', f'mse_{os.path.basename(model_pool_path).split("_")[-1]}'), rew_mse)
        np.save(os.path.join(policy_exp_details.results_dir, 'models', f'overall_mse_{os.path.basename(model_pool_path).split("_")[-1]}'), overall_mse)

if __name__ == '__main__':
    args = parse_args()
    policy_experiment = args.policy_experiment
    env_name = args.env_name

    # Manually specify parameters when debugging
    # policy_experiment = 'HO265'
    # env_name = 'Hopper'

    score_pool(policy_experiment, env_name)
