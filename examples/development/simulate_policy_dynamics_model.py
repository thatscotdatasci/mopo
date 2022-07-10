import argparse
import json
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from mujoco_py import MujocoException
from sklearn.decomposition import PCA

import mopo.static
from mopo.models.bnn import BNN
from mopo.models.fake_env import FakeEnv
from mopo.off_policy.loader import restore_pool_contiguous
from softlearning.environments.utils import get_environment_from_params
from softlearning.policies.utils import get_policy_from_variant
from softlearning.replay_pools.utils import get_replay_pool_from_variant
from softlearning.replay_pools import SimpleReplayPool
from softlearning.samplers.utils import get_sampler_from_variant
from dogo.results import get_experiment_details
from dogo.rollouts.collectors import RolloutCollector, MopoRolloutCollector


# NOTE: Items to be aware of:
# - the step method in FakeEnv can be run in deterministic mode, or not
# - there are parameters in bnn_params, however these should mostly not be used
# - choosing to be deterministic or not has a large impact
# - parameter deterministic is set to False in trining, which impacts the dynamics
# - parameter eval_deterministic is True in training, which impacts the policy

PARAMETERS_PATH = "/home/ajc348/rds/hpc-work/mopo/dogo/bnn_params.json"
EPISODE_LENGTH = 1000
DETERMINISTIC_MODEL = True
DETERMINISTIC_POLICY = True
START_LOCS_FROM_POLICY_TRAINING = True

PCA_1D = 'pca/pca_1d.pkl'
PCA_2D = 'pca/pca_2d.pkl'

assert EPISODE_LENGTH <= 1000

cols = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def plot_cumulative_reward(rollouts, penalty_coeff):
    fig, ax = plt.subplots(1, 1, figsize=(20,10))

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
        ax.fill_between(np.arange(EPISODE_LENGTH), min_v, max_v, color=cols[i], alpha=0.5)

    ax.legend()

    plt.savefig('policy_cumulative_rewards.jpeg')

def plot_reward(rollouts, penalty_coeff):
    fig, ax = plt.subplots(1, 1, figsize=(20,10))

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
        ax.fill_between(np.arange(EPISODE_LENGTH), min_v, max_v, color=cols[i], alpha=0.5)

    ax.legend()

    plt.savefig('policy_rewards.jpeg')

def plot_mse(rollouts, metric):
    fig, ax = plt.subplots(1, 1, figsize=(20,10))

    n_rollouts = len(rollouts)

    fake_metric = np.stack([rollouts[i]['fake'][metric] for i in range(n_rollouts)], axis=-1)
    eval_metric = np.stack([rollouts[i]['eval'][metric] for i in range(n_rollouts)], axis=-1)

    mse = (fake_metric-eval_metric)**2
    mse_flat = mse.reshape((mse.shape[0],-1))
    mse_mean = mse_flat.mean(axis=-1)
    mse_max = mse_flat.max(axis=-1)

    ax.plot(mse_mean)
    ax.fill_between(np.arange(EPISODE_LENGTH), np.zeros_like(mse_max), mse_max, alpha=0.5)

    # ax.legend()

    plt.savefig(f'policy_{metric}_mse.jpeg')

def plot_gym_mse(rollouts, metric):
    fig, ax = plt.subplots(1, 1, figsize=(20,10))

    n_rollouts = len(rollouts)

    fake_metric = np.stack([rollouts[i]['fake'][metric] for i in range(n_rollouts)], axis=-1)
    gym_metric = np.stack([rollouts[i]['gym'][metric] for i in range(n_rollouts)], axis=-1)

    mse = (fake_metric-gym_metric)**2
    mse_flat = mse.reshape((mse.shape[0],-1))
    mse_mean = mse_flat.mean(axis=-1)
    mse_max = mse_flat.max(axis=-1)

    ax.plot(mse_mean)
    ax.fill_between(np.arange(EPISODE_LENGTH), np.zeros_like(mse_max), mse_max, alpha=0.5)

    # ax.legend()

    plt.savefig(f'policy_gym_mse_{metric}.jpeg')

def plot_gym_cos(rollouts, metric):
    fig, ax = plt.subplots(1, 1, figsize=(20,10))

    n_rollouts = len(rollouts)

    fake_metric = np.stack([rollouts[i]['fake'][metric] for i in range(n_rollouts)], axis=-1)
    gym_metric = np.stack([rollouts[i]['gym'][metric] for i in range(n_rollouts)], axis=-1)

    cos = np.einsum('ijk,ijk->ik', fake_metric, gym_metric)/(np.linalg.norm(fake_metric, axis=1) * np.linalg.norm(gym_metric, axis=1))
    cos_mean = cos.mean(axis=-1)
    cos_min = cos.min(axis=-1)
    cos_max = cos.max(axis=-1)

    ax.plot(cos_mean)
    ax.fill_between(np.arange(EPISODE_LENGTH), cos_min, cos_max, alpha=0.5)

    # ax.legend()

    plt.savefig(f'policy_gym_cos_{metric}.jpeg')

def plot_visitation_landscape(rollouts):
    with open(PCA_2D, 'rb') as f:
        pca_2d = pickle.load(f)

    n_rollouts = len(rollouts)
    
    fake_obs = np.vstack([rollouts[i]['fake']['obs'] for i in range(n_rollouts)])
    fake_acts = np.vstack([rollouts[i]['fake']['acts'] for i in range(n_rollouts)])
    
    gym_obs = np.vstack([rollouts[i]['gym']['obs'] for i in range(n_rollouts)])
    gym_acts = np.vstack([rollouts[i]['gym']['acts'] for i in range(n_rollouts)])

    fake_pca = pca_2d.transform(np.hstack((fake_obs, fake_acts)))
    gym_pca = pca_2d.transform(np.hstack((gym_obs, gym_acts)))

    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    ax.scatter(fake_pca[:,0], fake_pca[:,1], marker='x', s=10, label='Model')
    ax.scatter(gym_pca[:,0], gym_pca[:,1], marker='x', s=10, label='Real Environment')
    ax.legend()
    plt.savefig(f'policy_s_a_pca_2d.jpeg')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy-experiment',
                        type=str,
                        help='Experiment whose dynamics model should be used.')
    parser.add_argument('--dynamics-experiment',
                        type=str,
                        help='Experiment whose dynamics model should be used.')
    parser.add_argument('--penalty-coeff',
                        type=float,
                        default=0.0,
                        help='The MOPO penalty coefficient.')
    # parser.add_argument('--max-path-length', '-l', type=int, default=1000)
    parser.add_argument('--num-rollouts', '-n', type=int, default=10)

    args = parser.parse_args()

    return args

def get_qpos_qvel(obs):
    # Return qpos and qvel values from the current obs
    # These are necessary if setting the environment
    qpos = np.hstack((np.zeros(1),obs[:8]))
    qvel = obs[8:]
    return qpos, qvel

def rollout_model(policy, fake_env, eval_env, gym_env, sampler):
    # Instantiate collector objects
    fake_collector = MopoRolloutCollector()
    eval_collector = RolloutCollector()
    gym_collector = RolloutCollector()

    # If a sampler is provided then sample from this
    # Otherwise, sample from the gym environment
    if sampler is not None:
        batch = sampler.random_batch(1)
        obs = batch['observations']

        obs_gym = obs
        gym_env._env.set_state(*get_qpos_qvel(obs_gym.flatten()))
    else:
        obs_gym = gym_env.convert_to_active_observation(gym_env.reset())[None,:]
        obs = obs_gym

    # Some policies can lead to instability in the real MuJoCo environment
    # If this occurs, stop using the real environment
    real_env_broken = False

    for _ in range(EPISODE_LENGTH):
        # Query the policy for the action to take
        act = policy.actions_np(obs)

        # Dynamics model - predicted next_obs and reward
        next_obs, rew, _, info = fake_env.step(obs, act, deterministic=DETERMINISTIC_MODEL)
        fake_collector.add_transition(obs, act, next_obs, rew, info['unpenalized_rewards'], info['penalty'])

        # Evaluation environment - set the state to the current state, apply action, get true next obs and reward
        next_obs_real = np.ones_like(obs)*np.nan
        rew_real = np.nan
        if not real_env_broken:
            eval_env._env.set_state(*get_qpos_qvel(obs.flatten()))
            try:
                next_obs_real, rew_real, _, _ = eval_env.step(act)
            except MujocoException:
                print('MuJoCo env became unstable')
                real_env_broken = True
        eval_collector.add_transition(obs, act, next_obs_real, rew_real)

        # Gym environment - follow the real environment dynamics, taking actions from the policy
        act_gym = policy.actions_np(obs_gym)
        next_obs_gym, rew_gym, _, _ = gym_env.step(act_gym)
        gym_collector.add_transition(obs_gym, act_gym, next_obs_gym, rew_gym)
        
        # Update the current observation for the next loop
        obs = next_obs
        obs_gym = next_obs_gym['observations'][None,:]
    
    # Return the populated collectors
    return {
        'fake': fake_collector.return_transitions(),
        'eval': eval_collector.return_transitions(),
        'gym':  gym_collector.return_transitions()
    }

def generate_rollouts(n_rollouts, policy, fake_env, eval_env, gym_env, sampler):
    # Create the desired number of rollouts
    rollouts = [rollout_model(policy, fake_env, eval_env, gym_env, sampler) for _ in range(n_rollouts)]
    return rollouts

def simulate_policy(args):
    session = tf.keras.backend.get_session()

    #######################
    # Load the policy model
    # Create a FakeEnv
    #######################
    policy_exp_details = get_experiment_details(args.policy_experiment, get_elites=False)
    policy_experiment_path = policy_exp_details.results_dir
    policy_checkpoint_path = os.path.join(
        policy_exp_details.results_dir,
        'ray_mopo',
        policy_exp_details.environment,
        policy_exp_details.base_dir,
        policy_exp_details.experiment_dir,
        'checkpoint_501',
    )

    policy_exp_params_path = os.path.join(policy_experiment_path, 'params.json')
    with open(policy_exp_params_path, 'r') as f:
        policy_exp_params = json.load(f)

    with session.as_default():
        pickle_path = os.path.join(policy_checkpoint_path, 'checkpoint.pkl')
        with open(pickle_path, 'rb') as f:
            picklable = pickle.load(f)

    policy = (
        get_policy_from_variant(policy_exp_params, eval_env, Qs=[None]))
    policy.set_weights(picklable['policy_weights'])

    ######################################################
    # Load evaluation and gym environments
    # Note: the eval environment is also a gym environment
    ######################################################
    environment_params = (
        policy_exp_params['environment_params']['evaluation']
        if 'evaluation' in policy_exp_params['environment_params']
        else policy_exp_params['environment_params']['training'])
    eval_env = get_environment_from_params(environment_params)
    gym_env = get_environment_from_params(environment_params)

    #########################
    # Load the dynamics model
    #########################
    dynamics_exp_details = get_experiment_details(args.dynamics_experiment, get_elites=True)
    dynamics_model_dir = os.path.join(dynamics_exp_details.results_dir, 'models')
    with open(PARAMETERS_PATH, 'r') as f:
        dynamics_params = json.load(f)
    dynamics_params["model_dir"] = dynamics_model_dir

    dynamics_model = BNN(dynamics_params)
    dynamics_model.model_loaded = True
    dynamics_model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})
    dynamics_model._model_inds = dynamics_exp_details.elites

    domain = environment_params['domain']
    static_fns = mopo.static[domain.lower()]
    fake_env = FakeEnv(
        dynamics_model, static_fns, penalty_coeff=args.penalty_coeff, penalty_learned_var=True
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
    with policy.set_deterministic(DETERMINISTIC_POLICY):
        rollouts = generate_rollouts(
            args.num_rollouts,
            policy,
            fake_env,
            eval_env,
            gym_env,
            sampler,
        )

    ###############
    # Print rewards
    ###############
    fake_rewards = [ro['fake']['rewards'].sum() for ro in rollouts]
    eval_rewards = [ro['eval']['rewards'].sum() for ro in rollouts]
    gym_rewards = [ro['gym']['rewards'].sum() for ro in rollouts]
    print('Fake Rewards: {}'.format(fake_rewards))
    print('Eval Rewards: {}'.format(eval_rewards))
    print('Gym Rewards: {}'.format(gym_rewards))
    print('---')
    print('Fake Mean: {}'.format(np.mean(fake_rewards)))
    print('Eval Mean: {}'.format(np.mean(eval_rewards)))
    print('Gym Mean: {}'.format(np.mean(gym_rewards)))
    
    return rollouts

if __name__ == '__main__':
    args = parse_args()
    rollouts = simulate_policy(args)
    plot_visitation_landscape(rollouts)
    plot_cumulative_reward(rollouts, args.penalty_coeff)
    plot_reward(rollouts, args.penalty_coeff)
    plot_mse(rollouts, 'next_obs')
    plot_mse(rollouts, 'rewards')
    plot_gym_mse(rollouts, 'obs')
    plot_gym_mse(rollouts, 'rewards')
    plot_gym_cos(rollouts, 'obs')
    plot_gym_cos(rollouts, 'rewards')

    # n_rollouts = len(rollouts)

    # obs_fake = np.stack([np.cumsum(rollouts[i]['fake']['obs']) for i in range(n_rollouts)], axis=-1)
    # np.save('policy_obs_fake.npy', obs_fake)

    # obs_gym = np.stack([np.cumsum(rollouts[i]['gym']['obs']) for i in range(n_rollouts)], axis=-1)
    # np.save('policy_obs_gym.npy', obs_fake)
