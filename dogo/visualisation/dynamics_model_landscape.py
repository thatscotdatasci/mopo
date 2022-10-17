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
from dogo.constants import HC_STATE_DIMS, HC_ACTION_DIMS
from dogo.results import get_experiment_details
from dogo.rollouts.collectors import RolloutCollector, MopoRolloutCollector


# NOTE: Items to be aware of:
# - the step method in FakeEnv can be run in deterministic mode, or not
# - there are parameters in bnn_params, however these should mostly not be used
# - choosing to be deterministic or not has a large impact
# - parameter deterministic is set to False in trining, which impacts the dynamics

PARAMETERS_PATH = os.path.expanduser("~/rds/hpc-work/mopo/dogo/bnn_params.json")
DATA_DIR = os.path.expanduser("~/rds/rds-dsk-lab-eWkDxBhxBrQ/dimorl/code/dogo_results/data")
OUTPUT_DIR = os.path.expanduser("~/rds/rds-dsk-lab-eWkDxBhxBrQ/dimorl/code/dogo_results/mopo/analysis/dynamics")
DETERMINISTIC_MODEL = True
N_RECORDS = 10000

PCA_1D = 'pca/pca_1d.pkl'

cols = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def plot_reward_landscape_2d(pca_1d, transitions, metric, red_op=None, red_axis=-1):
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    
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
    parser.add_argument('--dataset',
                        type=str,
                        default=None,
                        help='Dataset to load.')

    args = parser.parse_args()

    return args

def get_qpos_qvel(obs):
    # Return qpos and qvel values from the current obs
    # These are necessary if setting the environment
    qpos = np.hstack((np.zeros(1),obs[:8]))
    qvel = obs[8:]
    return qpos, qvel

def sample_transitions(args):

    #########################
    # Load the dynamics model
    # Create a FakeEnv
    #########################
    dynamics_exp_details = get_experiment_details(args.dynamics_experiment, get_elites=True)
    dynamics_model_dir = os.path.join(dynamics_exp_details.results_dir, 'models')
    with open(PARAMETERS_PATH, 'r') as f:
        dynamics_exp_params = json.load(f)
    dynamics_exp_params["model_dir"] = dynamics_model_dir

    dynamics_model = BNN(dynamics_exp_params)
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
        dynamics_model, static_fns, penalty_coeff=1.0, penalty_learned_var=True
    )

    ##################################
    # Create an evaluation environment
    ##################################
    eval_env = get_environment_from_params(environment_params)

    if args.dataset is not None:
        sa_arr = np.load(os.path.join(DATA_DIR, f'{args.dataset}.npy'))[:N_RECORDS,:HC_STATE_DIMS+HC_ACTION_DIMS]
    else:
        sa_arr = None

    ####################
    # Create transitions
    ####################
    fake_collector = MopoRolloutCollector()
    eval_collector = RolloutCollector()

    for i in range(N_RECORDS):
        if sa_arr is not None:
            obs = sa_arr[i,:HC_STATE_DIMS]
            act = sa_arr[i,HC_STATE_DIMS:HC_STATE_DIMS+HC_ACTION_DIMS]
            eval_env._env.set_state(*get_qpos_qvel(obs.flatten()))
        else:
            obs = eval_env.reset()['observations']

        next_obs, rew, _, info = fake_env.step(obs, act, deterministic=DETERMINISTIC_MODEL)
        fake_collector.add_transition(obs, act, next_obs, rew, info)

        next_obs_real, rew_real, _, _ = eval_env.step(act)
        eval_collector.add_transition(obs, act, next_obs_real, rew_real)
    
    return {
        'fake': fake_collector.return_transitions(),
        'eval': eval_collector.return_transitions(),
    }, sa_arr

def get_file_prefix(args):
    return f'{args.dynamics_experiment}_{args.dataset}'

def save_state_action(file_prefix, state_action_arr):
    np.save(os.path.join(OUTPUT_DIR, f'{file_prefix}_state_action.npy'), state_action_arr)

def save_metric(file_prefix, transitions, metric):
    np.save(os.path.join(OUTPUT_DIR, f'{file_prefix}_{metric}.npy'), transitions['fake'][metric])

def save_rewards_mse(file_prefix, transitions):
    fake_rews = transitions['fake']['unpen_rewards']
    true_rews = transitions['eval']['rewards']
    rews_mse = ((fake_rews-true_rews)**2)
    np.save(os.path.join(OUTPUT_DIR, f'{file_prefix}_rews_mse.npy'), rews_mse)

if __name__ == '__main__':
    args = parse_args()
    transitions, sa_arr = sample_transitions(args)

    file_prefix = get_file_prefix(args)

    save_state_action(file_prefix, sa_arr)

    save_rewards_mse(file_prefix, transitions)

    # save_metric(file_prefix, transitions, 'rewards')
    save_metric(file_prefix, transitions, 'unpen_rewards')
    save_metric(file_prefix, transitions, 'reward_pens')
    # save_metric(file_prefix, transitions, 'ensemble_means_mean')
    save_metric(file_prefix, transitions, 'ensemble_means_std')
    save_metric(file_prefix, transitions, 'ensemble_vars_mean')
    save_metric(file_prefix, transitions, 'ensemble_vars_std')
    save_metric(file_prefix, transitions, 'ensemble_stds_norm')

    # with open(PCA_1D, 'rb') as f:
    #     pca_1d = pickle.load(f)

    # plot_reward_landscape_2d(pca_1d, transitions, 'rewards')
    # plot_reward_landscape_2d(pca_1d, transitions, 'ensemble_means_std', red_op='mean')
    # plot_reward_landscape_2d(pca_1d, transitions, 'ensemble_vars_max', red_op='max')

