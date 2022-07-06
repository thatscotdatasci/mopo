import argparse
from distutils.util import strtobool
import json
import os
import pickle

import numpy as np
import tensorflow as tf
from mujoco_py import MujocoException

import mopo.static
from mopo.models.bnn import BNN
from mopo.models.fake_env import FakeEnv
from mopo.off_policy.loader import restore_pool_contiguous
from softlearning.environments.utils import get_environment_from_params
from softlearning.policies.utils import get_policy_from_variant
from softlearning.replay_pools.utils import get_replay_pool_from_variant
from dogo.results import get_experiment_details
from softlearning.replay_pools import SimpleReplayPool
from softlearning.samplers.utils import get_sampler_from_variant


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
START_LOCS_FROM_POLICY_TRAINING = False

class RolloutCollector:
    def __init__(self) -> None:
        self.obs = []
        self.acts = []
        self.next_obs = []
        self.rews = []

    def add_transition(self, obs, act, next_obs, rew):
        if type(obs) == dict:
            obs = obs['observations']
        if type(next_obs) == dict:
            next_obs = next_obs['observations']
        if type(rew) == np.array:
            rew = rew[0,0]

        self.obs.append(obs)
        self.acts.append(act)
        self.next_obs.append(next_obs)
        self.rews.append(rew)

    def return_transitions(self):
        return {
            'obs':      np.vstack(self.obs),
            'acts':     np.vstack(self.acts),
            'next_obs': np.vstack(self.next_obs),
            'rewards':  np.vstack(self.rews),
        }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('policy_experiment',
                        type=str,
                        help='Experiment whose dynamics model should be used.')
    parser.add_argument('dynamics_experiment',
                        type=str,
                        help='Experiment whose dynamics model should be used.')
    # parser.add_argument('--max-path-length', '-l', type=int, default=1000)
    parser.add_argument('--num-rollouts', '-n', type=int, default=10)
    parser.add_argument('--render-mode', '-r',
                        type=str,
                        default=None,
                        choices=('human', 'rgb_array', 'None'),
                        help="Mode to render the rollouts in.")

    args = parser.parse_args()
    
    if args.render_mode == 'None':
        args.render_mode = None

    return args

def get_qpos_qvel(obs):
    qpos = np.hstack((np.zeros(1),obs[:8]))
    qvel = obs[8:]
    return qpos, qvel

def rollout_model(policy, fake_env, eval_env, gym_env, sampler):
    fake_collector = RolloutCollector()
    eval_collector = RolloutCollector()
    gym_collector = RolloutCollector()

    if sampler is not None:
        batch = sampler.random_batch(1)
        obs = batch['observations']

        obs_gym = obs
        gym_env._env.set_state(*get_qpos_qvel(obs_gym.flatten()))
    else:
        obs_gym = gym_env.convert_to_active_observation(gym_env.reset())[None,:]
        obs = obs_gym

    real_env_broken = False
    for _ in range(EPISODE_LENGTH):
        # Query policy for the action to take
        act = policy.actions_np(obs)

        # Dynamics model - predicted next_obs and reward
        next_obs, rew, _, _ = fake_env.step(obs, act, deterministic=DETERMINISTIC_MODEL)
        fake_collector.add_transition(obs, act, next_obs, rew)

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

        # Gym environment - follow the real dynamics, taking actions from the policy
        act_gym = policy.actions_np(obs_gym)
        next_obs_gym, rew_gym, _, _ = gym_env.step(act_gym)
        gym_collector.add_transition(obs_gym, act_gym, next_obs_gym, rew_gym)
        
        # Update the current observation for the next loop
        obs = next_obs
        obs_gym = next_obs_gym['observations'][None,:]
    
    return {
        'fake': fake_collector.return_transitions(),
        'eval': eval_collector.return_transitions(),
        'gym':  gym_collector.return_transitions()
    }

def generate_rollouts(n_rollouts, policy, fake_env, eval_env, gym_env, sampler):
    rollouts = [rollout_model(policy, fake_env, eval_env, gym_env, sampler) for _ in range(n_rollouts)]
    return rollouts

def simulate_policy(args):
    session = tf.keras.backend.get_session()

    #######################
    # Load the policy model
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

    variant_path = os.path.join(policy_experiment_path, 'params.json')
    with open(variant_path, 'r') as f:
        variant = json.load(f)

    with session.as_default():
        pickle_path = os.path.join(policy_checkpoint_path, 'checkpoint.pkl')
        with open(pickle_path, 'rb') as f:
            picklable = pickle.load(f)

    environment_params = (
        variant['environment_params']['evaluation']
        if 'evaluation' in variant['environment_params']
        else variant['environment_params']['training'])
    eval_env = get_environment_from_params(environment_params)
    gym_env = get_environment_from_params(environment_params)

    policy = (
        get_policy_from_variant(variant, eval_env, Qs=[None]))
    policy.set_weights(picklable['policy_weights'])

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
        dynamics_model, static_fns, penalty_coeff=0., penalty_learned_var=True
    )

    #####################################
    # Create and load replay pool/dataset
    #####################################

    if START_LOCS_FROM_POLICY_TRAINING:
        replay_pool = get_replay_pool_from_variant(variant, eval_env)
        restore_pool_contiguous(replay_pool, variant['algorithm_params']['kwargs']['pool_load_path'])

        sampler = get_sampler_from_variant(variant)
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
    simulate_policy(args)
