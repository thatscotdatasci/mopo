import argparse
import json
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.decomposition import PCA

import mopo.static
from mopo.models.bnn import BNN
from mopo.models.fake_env import FakeEnv
from mopo.off_policy.loader import restore_pool_contiguous
from softlearning.environments.utils import get_environment_from_params
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

PARAMETERS_PATH = "/home/ajc348/rds/hpc-work/mopo/dogo/bnn_params.json"
DETERMINISTIC_MODEL = True
START_LOCS_FROM_POLICY_TRAINING = True

PCA_1D = 'pca/pca_1d.pkl'
PCA_2D = 'pca/pca_2d.pkl'

cols = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def plot_reward_landscape_2d(transitions, metric, red_op=None, red_axis=-1):
    fig, ax = plt.subplots(1, 1, figsize=(10,10))

    with open(PCA_1D, 'rb') as f:
        pca_1d = pickle.load(f)
    
    obs = transitions['fake']['obs']
    acts = transitions['fake']['acts']
    obs_acts_pca = pca_1d.transform(np.hstack((obs, acts)))
    
    fake_metric = transitions['fake'][metric]
    if red_op is not None:
        fake_metric = getattr(fake_metric, red_op)(axis=red_axis)
    ax.scatter(obs_acts_pca, fake_metric, marker='x', s=10, label='Model')
    
    if metric in transitions['eval']:
        eval_metric = transitions['eval'][metric]
        if red_op is not None:
            eval_metric = getattr(eval_metric, red_op)(axis=red_axis)
        ax.scatter(obs_acts_pca, eval_metric, marker='x', s=10, label='Real Environment')
    
    ax.legend()
    plt.savefig(f'dynamics_{metric}_1d.jpeg')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dynamics-experiment',
                        type=str,
                        help='Experiment whose dynamics model should be used.')
    parser.add_argument('--penalty-coeff',
                        type=float,
                        default=0.0,
                        help='The MOPO penalty coefficient.')
    parser.add_argument('--render-mode', '-r',
                        type=str,
                        default=None,
                        choices=('human', 'rgb_array', 'None'),
                        help="Mode to render the rollouts in.")

    args = parser.parse_args()
    
    if args.render_mode == 'None':
        args.render_mode = None

    return args

def sample_transitions(args):

    #########################
    # Load the dynamics model
    # Create a FakeEnv
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

    dynamics_exp_params_path = os.path.join(dynamics_exp_details.results_dir, 'params.json')
    with open(dynamics_exp_params_path, 'r') as f:
        dynamics_exp_params = json.load(f)

    environment_params = (
        dynamics_exp_params['environment_params']['evaluation']
        if 'evaluation' in dynamics_exp_params['environment_params']
        else dynamics_exp_params['environment_params']['training']
    )
    domain = environment_params['domain']
    static_fns = mopo.static[domain.lower()]
    fake_env = FakeEnv(
        dynamics_model, static_fns, penalty_coeff=args.penalty_coeff, penalty_learned_var=True
    )

    ##################################
    # Create an evaluation environment
    ##################################
    eval_env = get_environment_from_params(environment_params)

    ####################
    # Create transitions
    ####################
    fake_collector = MopoRolloutCollector()
    eval_collector = RolloutCollector()

    for _ in range(10000):
        obs = eval_env.reset()['observations']
        act = eval_env.action_space.sample()

        next_obs, rew, _, info = fake_env.step(obs, act, deterministic=DETERMINISTIC_MODEL)
        fake_collector.add_transition(obs, act, next_obs, rew, info)

        next_obs_real, rew_real, _, _ = eval_env.step(act)
        eval_collector.add_transition(obs, act, next_obs_real, rew_real)
    
    return {
        'fake': fake_collector.return_transitions(),
        'eval': eval_collector.return_transitions(),
    }

if __name__ == '__main__':
    args = parse_args()
    transitions = sample_transitions(args)
    plot_reward_landscape_2d(transitions, 'rewards')
    plot_reward_landscape_2d(transitions, 'ensemble_means_std', red_op='mean')
    plot_reward_landscape_2d(transitions, 'ensemble_vars_max', red_op='max')

