import argparse
import json
import os
import pickle
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from mujoco_py import MujocoException

import mopo.static
from mopo.models.bnn import BNN
from mopo.models.fake_env import FakeEnv
from mopo.off_policy.loader import restore_pool_contiguous
from softlearning.environments.utils import get_environment_from_params
from softlearning.policies.utils import get_policy_from_variant
from softlearning.replay_pools.utils import get_replay_pool_from_variant
from softlearning.samplers.utils import get_sampler_from_variant
from dogo.results import get_experiment_details
from dogo.rollouts.collectors import RolloutCollector, MopoRolloutCollector


#####################################################################################
# Simulate a policy within the real environment, or against a learned dynamics
# model.
#####################################################################################
# NOTE: Important items to be aware of:
# - the `step` method in FakeEnv can be run in deterministic or stochastic mode
# - unsurprisingly, the mode chosen has an impact on the results
# - there are parameters in bnn_params, however these are not used
#   o they are simply necessary to instantiate a BNN oject
# - parameter deterministic is set to False in trining, which impacts the dynamics
# - parameter eval_deterministic is True in training, which impacts the policy
# - by default, both the dynamics and policy are run in deterministic mode
# - running the dynamics in deterministic mode does not impact the calculation of
#   the MOPO penalty - the standard deviations of the predictive dists are still used
# - additionally note that rollout starting locations can be drawn either from
#   the environment's initial state distribution, or from the data used to train the
#   dynamics model. This *should* default to the real initial state distribution
#####################################################################################


# BNN parameters paths
# As stated above, these are needed to instantiate a `BNN` object - they should not impact scoring
PARAMETERS_PATH_HC = os.path.expanduser("~/rds/hpc-work/mopo/dogo/bnn_params_halfcheetah.json")
PARAMETERS_PATH_H = os.path.expanduser("~/rds/hpc-work/mopo/dogo/bnn_params_hopper.json")
PARAMETERS_PATH_W = os.path.expanduser("~/rds/hpc-work/mopo/dogo/bnn_params_walker2d.json")

OUTPUT_DIR = os.path.expanduser("~/rds/rds-dsk-lab-eWkDxBhxBrQ/dimorl/code/dogo_results/mopo/analysis/policy")

# The HalfCheetah environment used has an upper episode length limit of 1000 steps
MAX_EPISODE_LENGTH = 1000
assert MAX_EPISODE_LENGTH <= 1000

# Whether to draw episode starting locations from the dataset used to train the dynamics models
START_LOCS_FROM_POLICY_TRAINING = False

PCA_1D = 'pca/pca_1d.pkl'
PCA_2D = 'pca/pca_2d.pkl'

cols = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def plot_cumulative_reward(rollouts, penalty_coeff):
    ######################
    # Not currently in use
    ######################
    _, ax = plt.subplots(1, 1, figsize=(20,10))

    n_rollouts = len(rollouts)
    fake_penalties = np.stack([np.cumsum(rollouts[i]['fake']['reward_pens']) for i in range(n_rollouts)], axis=-1)
    fake_unpen_rewards = np.stack([np.cumsum(rollouts[i]['fake']['unpen_rewards']) for i in range(n_rollouts)], axis=-1)
    fake_pen_rewards = np.stack([np.cumsum(rollouts[i]['fake']['rewards']) for i in range(n_rollouts)], axis=-1)
    eval_rewards = np.stack([np.cumsum(rollouts[i]['eval']['rewards']) for i in range(n_rollouts)], axis=-1)
    gym_rewards  = np.stack([np.cumsum(rollouts[i]['gym']['rewards']) for i in range(n_rollouts)], axis=-1)
    
    for i, (metric, label) in enumerate([
        (fake_penalties, 'Reward Penalty'),
        (fake_unpen_rewards, 'Unpenalised Prediction'),
        (fake_pen_rewards, f'Penalised Prediction (coeff: {penalty_coeff})'),
        (eval_rewards, 'True Value'),
        (gym_rewards, 'Real Environment'),
    ]):
        mean = metric.mean(axis=-1)
        min_v = metric.min(axis=-1)
        max_v = metric.max(axis=-1)

        ax.plot(mean, c=cols[i], label=label)
        ax.fill_between(np.arange(MAX_EPISODE_LENGTH), min_v, max_v, color=cols[i], alpha=0.5)

    ax.legend()

    plt.savefig('policy_cumulative_rewards.jpeg')

def plot_reward(rollouts, penalty_coeff):
    ######################
    # Not currently in use
    ######################
    _, ax = plt.subplots(1, 1, figsize=(20,10))

    n_rollouts = len(rollouts)
    penalties = np.stack([rollouts[i]['fake']['reward_pens'].flatten() for i in range(n_rollouts)], axis=-1)
    pen_rewards = np.stack([rollouts[i]['fake']['rewards'].flatten() for i in range(n_rollouts)], axis=-1)
    unpen_rewards = np.stack([rollouts[i]['fake']['unpen_rewards'].flatten() for i in range(n_rollouts)], axis=-1)

    for i, (metric, label) in enumerate([
        (penalties, 'Penalty'),
        (pen_rewards, f'Penalised Prediction (coeff: {penalty_coeff})'),
        (unpen_rewards, 'Unpenalised Reward'),
    ]):
        mean = metric.mean(axis=-1)
        min_v = metric.min(axis=-1)
        max_v = metric.max(axis=-1)

        ax.plot(mean, c=cols[i], label=label)
        ax.fill_between(np.arange(MAX_EPISODE_LENGTH), min_v, max_v, color=cols[i], alpha=0.5)

    ax.legend()

    plt.savefig('policy_rewards.jpeg')

def plot_mse(rollouts, metric):
    ######################
    # Not currently in use
    ######################
    _, ax = plt.subplots(1, 1, figsize=(20,10))

    n_rollouts = len(rollouts)

    fake_metric = np.stack([rollouts[i]['fake'][metric] for i in range(n_rollouts)], axis=-1)
    eval_metric = np.stack([rollouts[i]['eval'][metric] for i in range(n_rollouts)], axis=-1)

    mse = (fake_metric-eval_metric)**2
    mse_flat = mse.reshape((mse.shape[0],-1))
    mse_mean = mse_flat.mean(axis=-1)
    mse_min = mse_flat.min(axis=-1)
    mse_max = mse_flat.max(axis=-1)

    ax.plot(mse_mean)
    ax.fill_between(np.arange(MAX_EPISODE_LENGTH), mse_min, mse_max, alpha=0.5)

    plt.savefig(f'policy_{metric}_mse.jpeg')

def plot_gym_mse(rollouts, metric):
    ######################
    # Not currently in use
    ######################
    _, ax = plt.subplots(1, 1, figsize=(20,10))

    n_rollouts = len(rollouts)
    fake_metric = np.stack([rollouts[i]['fake'][metric] for i in range(n_rollouts)], axis=-1)
    gym_metric = np.stack([rollouts[i]['gym'][metric] for i in range(n_rollouts)], axis=-1)

    mse = (fake_metric-gym_metric)**2
    mse_flat = mse.reshape((mse.shape[0],-1))
    mse_mean = mse_flat.mean(axis=-1)
    mse_min = mse_flat.min(axis=-1)
    mse_max = mse_flat.max(axis=-1)

    ax.plot(mse_mean)
    ax.fill_between(np.arange(MAX_EPISODE_LENGTH), mse_min, mse_max, alpha=0.5)

    plt.savefig(f'policy_gym_mse_{metric}.jpeg')

def plot_gym_cos(rollouts, metric):
    """Cosine similarity"""
    ######################
    # Not currently in use
    ######################
    _, ax = plt.subplots(1, 1, figsize=(20,10))

    n_rollouts = len(rollouts)
    fake_metric = np.stack([rollouts[i]['fake'][metric] for i in range(n_rollouts)], axis=-1)
    gym_metric = np.stack([rollouts[i]['gym'][metric] for i in range(n_rollouts)], axis=-1)

    cos = np.einsum('ijk,ijk->ik', fake_metric, gym_metric)/(np.linalg.norm(fake_metric, axis=1) * np.linalg.norm(gym_metric, axis=1))
    cos_mean = cos.mean(axis=-1)
    cos_min = cos.min(axis=-1)
    cos_max = cos.max(axis=-1)

    ax.plot(cos_mean)
    ax.fill_between(np.arange(MAX_EPISODE_LENGTH), cos_min, cos_max, alpha=0.5)

    plt.savefig(f'policy_gym_cos_{metric}.jpeg')

def plot_visitation_landscape(rollouts):
    ######################
    # Not currently in use
    ######################
    with open(PCA_2D, 'rb') as f:
        pca_2d = pickle.load(f)

    n_rollouts = len(rollouts)
    
    fake_obs = np.vstack([rollouts[i]['fake']['obs'] for i in range(n_rollouts)])
    fake_acts = np.vstack([rollouts[i]['fake']['acts'] for i in range(n_rollouts)])
    
    gym_obs = np.vstack([rollouts[i]['gym']['obs'] for i in range(n_rollouts)])
    gym_acts = np.vstack([rollouts[i]['gym']['acts'] for i in range(n_rollouts)])

    fake_pca = pca_2d.transform(np.hstack((fake_obs, fake_acts)))
    gym_pca = pca_2d.transform(np.hstack((gym_obs, gym_acts)))

    _, ax = plt.subplots(1, 1, figsize=(10,10))
    ax.scatter(fake_pca[:,0], fake_pca[:,1], marker='x', s=10, label='Model')
    ax.scatter(gym_pca[:,0], gym_pca[:,1], marker='x', s=10, label='Real Environment')
    ax.legend()

    plt.savefig(f'policy_s_a_pca_2d.jpeg')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy-experiment',
                        type=str,
                        help='Experiment whose policy model should be used.')
    parser.add_argument('--dynamics-experiment',
                        type=str,
                        help='Experiment whose dynamics model should be used.')
    parser.add_argument('--stochastic-policy',
                        action='store_true',
                        help='Run stochastic policy.')
    parser.add_argument('--stochastic-model',
                        action='store_true',
                        help='Run stochastic model.')
    # parser.add_argument('--max-path-length', '-l', type=int, default=1000)
    parser.add_argument('--seed', '-s', type=int, help='Random seed to use')
    parser.add_argument('--num-rollouts', '-n', type=int, default=10, help='Number of rollouts to generate.')

    args = parser.parse_args()

    args.deterministic_policy = not args.stochastic_policy 
    args.deterministic_model = not args.stochastic_model

    return args

def get_qpos_qvel(obs, env_name):
    # Return qpos and qvel values from the current obs
    # These are necessary if setting the environment
    pos_vel_split_ind = 5 if env_name == 'Hopper' else 8
    qpos = np.hstack((np.zeros(1),obs[:pos_vel_split_ind]))
    qvel = obs[pos_vel_split_ind:]
    return qpos, qvel

def rollout_model(policy, fake_env, eval_env, gym_env, sampler, env_name, deterministic_model=True, init_obs_act=None):
    # Instantiate collector objects
    fake_collector = MopoRolloutCollector()
    eval_collector = RolloutCollector()
    gym_collector = RolloutCollector()

    # If a sampler is provided then sample from this
    # Otherwise, sample from the gym environment
    if init_obs_act is not None:
        init_obs, init_act = init_obs_act
        obs = obs_gym = init_obs.copy()
        gym_env._env.set_state(*get_qpos_qvel(obs_gym.flatten(), env_name))
    elif sampler is not None:
        batch = sampler.random_batch(1)
        obs = obs_gym = batch['observations']
        gym_env._env.set_state(*get_qpos_qvel(obs_gym.flatten(), env_name))

        init_obs = obs.copy()
        init_act = None
    else:
        obs_gym = obs = gym_env.convert_to_active_observation(gym_env.reset())[None,:]

        init_obs = obs.copy()
        init_act = None

    # Some policies can lead to instability in the real MuJoCo environment
    # If this occurs, stop using the real environment
    real_env_broken = False

    n_steps = 0
    done_fake = False
    done_gym = False

    # It might be that the episode finishes earlier in one environment than the others - keep running until they're all done
    while not all([done_fake, done_gym]) and n_steps < MAX_EPISODE_LENGTH:
        # Query the policy for the action to take, unless in the first step and an initial action is provided
        if n_steps == 0:
            if init_act is not None:
                act = init_act
            else:
                act = policy.actions_np(obs)
                init_act = act.copy()
        else:
            act = policy.actions_np(obs)

        # Dynamics model - predicted next_obs and reward
        next_obs = np.ones_like(obs)*np.nan
        rew = np.nan
        if not done_fake:
            next_obs, rew, done_fake, info = fake_env.step(obs, act, deterministic=deterministic_model)
            done_fake = done_fake.item()
        fake_collector.add_transition(obs, act, next_obs, rew, info)

        # Evaluation environment - set the state to the current state, apply action, get true next obs and reward
        next_obs_real = np.ones_like(obs)*np.nan
        rew_real = np.nan
        if not done_fake or not real_env_broken:
            eval_env._env.set_state(*get_qpos_qvel(obs.flatten(), env_name))
            try:
                next_obs_real, rew_real, _, _ = eval_env.step(act)
            except MujocoException:
                print('MuJoCo env became unstable')
                real_env_broken = True
        eval_collector.add_transition(obs, act, next_obs_real, rew_real)

        # Gym environment - follow the real environment dynamics, taking actions from the policy
        next_obs_gym = {'observations': np.ones_like(obs)*np.nan}
        rew_gym = np.nan
        if not done_gym:
            act_gym = policy.actions_np(obs_gym)
            next_obs_gym, rew_gym, done_gym, _ = gym_env.step(act_gym)
        else:
            act_gym = np.ones_like(act)*np.nan
        gym_collector.add_transition(obs_gym, act_gym, next_obs_gym, rew_gym)
        
        # Update the current observation for the next loop
        obs = next_obs
        obs_gym = next_obs_gym['observations'][None,:]

        # Update the step counter
        n_steps += 1
    
    # Return the populated collectors
    return {
        'fake': fake_collector.return_transitions(),
        'eval': eval_collector.return_transitions(),
        'gym':  gym_collector.return_transitions()
    }, (init_obs, init_act)

def generate_rollouts(n_rollouts, policy, fake_env, eval_env, gym_env, sampler, env_name, deterministic_model, q_value_estimate=True):
    # Create the desired number of rollouts
    if q_value_estimate:
        init_rollout, init_obs_act = rollout_model(policy, fake_env, eval_env, gym_env, sampler, env_name, deterministic_model)
        rollouts = [init_rollout]
        for _ in range(n_rollouts-1):
            rollouts.append(rollout_model(policy, fake_env, eval_env, gym_env, sampler, env_name, deterministic_model, init_obs_act=init_obs_act)[0])
    else:
        rollouts = [rollout_model(policy, fake_env, eval_env, gym_env, sampler, env_name, deterministic_model)[0] for _ in range(n_rollouts)]
    return rollouts

def simulate_policy(args):
    np.random.seed(args.seed)
    tf.compat.v1.random.set_random_seed(args.seed)
    session = tf.keras.backend.get_session()

    #######################
    # Load the policy model
    # Create a FakeEnv
    #######################
    policy_exp_details = get_experiment_details(args.policy_experiment, get_elites=False)
    policy_experiment_path = policy_exp_details.results_dir
    policy_checkpoint_paths = glob(os.path.join(
        policy_exp_details.results_dir,
        'ray_mopo',
        policy_exp_details.environment,
        policy_exp_details.base_dir,
        policy_exp_details.experiment_dir,
        'checkpoint_*1',
    ))

    if len(policy_checkpoint_paths) != 1:
        raise RuntimeError(f'No final policy identified for {args.policy_experiment}')
    policy_checkpoint_path = policy_checkpoint_paths[0]

    policy_exp_params_path = os.path.join(policy_experiment_path, 'params.json')
    with open(policy_exp_params_path, 'r') as f:
        policy_exp_params = json.load(f)

    with session.as_default():
        pickle_path = os.path.join(policy_checkpoint_path, 'checkpoint.pkl')
        with open(pickle_path, 'rb') as f:
            picklable = pickle.load(f)

    environment_params = (
        policy_exp_params['environment_params']['evaluation']
        if 'evaluation' in policy_exp_params['environment_params']
        else policy_exp_params['environment_params']['training'])
    
    eval_env = get_environment_from_params(environment_params)
    eval_env.seed(args.seed)

    gym_env = get_environment_from_params(environment_params)
    gym_env.seed(args.seed)

    policy = (
        get_policy_from_variant(policy_exp_params, eval_env, Qs=[None]))
    policy.set_weights(picklable['policy_weights'])

    #########################
    # Load the dynamics model
    #########################
    dynamics_exp = args.dynamics_experiment or policy_exp_details.dynamics_model_exp
    dynamics_exp_details = get_experiment_details(dynamics_exp, get_elites=True)
    dynamics_model_dir = os.path.join(dynamics_exp_details.results_dir, 'models')

    if dynamics_exp_details.environment == 'HalfCheetah':
        parameters_path = PARAMETERS_PATH_HC
    elif dynamics_exp_details.environment == 'Hopper':
        parameters_path = PARAMETERS_PATH_H
    elif dynamics_exp_details.environment == 'Walker2d':
        parameters_path = PARAMETERS_PATH_HC
    else:
        raise RuntimeError(f'No parameters file for environment: {dynamics_exp_details.environment}')

    with open(parameters_path, 'r') as f:
        dynamics_params = json.load(f)
    dynamics_params["model_dir"] = dynamics_model_dir

    dynamics_model = BNN(dynamics_params)
    dynamics_model.model_loaded = True
    dynamics_model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})
    dynamics_model._model_inds = dynamics_exp_details.elites

    domain = environment_params['domain']
    static_fns = mopo.static[domain.lower()]
    fake_env = FakeEnv(
        dynamics_model, static_fns, penalty_coeff=1.0, penalty_learned_var=True
    )

    ############################################
    # Create and load replay pool/dataset
    # The pool used in policy training is loaded
    ############################################
    if START_LOCS_FROM_POLICY_TRAINING:
        replay_pool = get_replay_pool_from_variant(policy_exp_params, eval_env)
        restore_pool_contiguous(replay_pool, policy_exp_params['algorithm_params']['kwargs']['pool_load_path'])

        sampler = get_sampler_from_variant(policy_exp_params)
        sampler.initialize(eval_env, policy, replay_pool)
    else:
        sampler = None

    #################
    # Create rollouts
    #################
    with policy.set_deterministic(args.deterministic_policy):
        rollouts = generate_rollouts(
            args.num_rollouts,
            policy,
            fake_env,
            eval_env,
            gym_env,
            sampler,
            domain,
            args.deterministic_model
        )

    ###############
    # Print rewards
    ###############
    fake_pen_rewards = [ro['fake']['rewards'].sum() for ro in rollouts]
    fake_unpen_rewards = [ro['fake']['unpen_rewards'].sum() for ro in rollouts]
    eval_rewards = [ro['eval']['rewards'].sum() for ro in rollouts]
    gym_rewards = [ro['gym']['rewards'].sum() for ro in rollouts]
    print('---')
    print(f'Dynamics Model: {dynamics_exp} - Deterministic: {args.deterministic_model}')
    print(f'Policy: {args.policy_experiment} - Deterministic: {args.deterministic_policy}')
    print('---')
    print('Fake Pen Rewards: {}'.format(fake_pen_rewards))
    print('Fake Unpen Rewards: {}'.format(fake_unpen_rewards))
    print('Eval Rewards: {}'.format(eval_rewards))
    print('Gym Rewards: {}'.format(gym_rewards))
    print('---')
    print('Fake Pen Mean: {}'.format(np.nanmean(fake_pen_rewards)))
    print('Fake Unpen Mean: {}'.format(np.nanmean(fake_unpen_rewards)))
    print('Eval Mean: {}'.format(np.nanmean(eval_rewards)))
    print('Gym Mean: {}'.format(np.nanmean(gym_rewards)))

    file_prefix = get_file_prefix(args, dynamics_exp)
    
    return rollouts, file_prefix

def get_file_prefix(args, dynamics_exp):
    base_name = f'{dynamics_exp}_{args.policy_experiment}_dm{args.deterministic_model}_dp{args.deterministic_policy}'
    if args.deterministic_model or args.deterministic_policy:
        base_name += '_' + str(args.seed)
    return base_name

def stack_arrays(arr_list):
    arr_lens = [len(arr) for arr in arr_list]
    max_arr_len = max(arr_lens)
    return  np.stack([np.pad(arr, ((0,max_arr_len-len(arr)), (0,0)), mode='constant', constant_values=np.NaN) for arr in arr_list], axis=0)

def save_state_action(file_prefix, env, rollouts):
    obs_acts_arr_list = [np.hstack((
        ro[env]['obs'], ro[env]['acts']
    )) for ro in rollouts]
    state_action_arr = stack_arrays(obs_acts_arr_list)
    np.save(os.path.join(OUTPUT_DIR, f'{file_prefix}_{env}_state_action.npy'), state_action_arr)

    # with open(PCA_1D, 'rb') as f:
    #     pca_1d = pickle.load(f)
    # state_action_pca_1d = np.stack([pca_1d.transform(i) for i in state_action_arr], axis=-1)
    # np.save(os.path.join(OUTPUT_DIR, f'{file_prefix}_{env}_state_action_pca_1d.npy'), state_action_pca_1d)

    # with open(PCA_2D, 'rb') as f:
    #     pca_2d = pickle.load(f)
    # state_action_pca_2d = np.stack([pca_2d.transform(i) for i in state_action_arr], axis=-1)
    # np.save(os.path.join(OUTPUT_DIR, f'{file_prefix}_{env}_state_action_pca_2d.npy'), state_action_pca_2d)

def save_metric(file_prefix, rollouts, env, metric):
    np.save(
        os.path.join(
            OUTPUT_DIR,
            f'{file_prefix}_{env}_{metric}.npy'
            ),
       stack_arrays([ro[env][metric] for ro in rollouts])
    )

if __name__ == '__main__':
    args = parse_args()
    rollouts, file_prefix = simulate_policy(args)

    save_state_action(file_prefix, 'fake', rollouts)
    save_state_action(file_prefix, 'gym', rollouts)

    save_metric(file_prefix, rollouts, 'fake', 'reward_pens')
    save_metric(file_prefix, rollouts, 'fake', 'unpen_rewards')
    # save_metric(file_prefix, rollouts, 'fake', 'ensemble_means_std')
    # save_metric(file_prefix, rollouts, 'fake', 'ensemble_vars_mean')
    # save_metric(file_prefix, rollouts, 'fake', 'ensemble_vars_std')
    # save_metric(file_prefix, rollouts, 'fake', 'ensemble_stds_norm')

    save_metric(file_prefix, rollouts, 'eval', 'rewards')
    
    save_metric(file_prefix, rollouts, 'gym', 'rewards')

    # plot_visitation_landscape(rollouts)
    # plot_cumulative_reward(rollouts, args.penalty_coeff)
    # plot_reward(rollouts, args.penalty_coeff)
    # plot_mse(rollouts, 'next_obs')
    # plot_mse(rollouts, 'rewards')
    # plot_gym_mse(rollouts, 'obs')
    # plot_gym_mse(rollouts, 'rewards')
    # plot_gym_cos(rollouts, 'obs')
    # plot_gym_cos(rollouts, 'rewards')
