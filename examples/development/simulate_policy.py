import argparse
from distutils.util import strtobool
import json
import os
import pickle

import numpy as np
import tensorflow as tf
from dogo.results import get_experiment_details

from softlearning.environments.utils import get_environment_from_params
from softlearning.policies.utils import get_policy_from_variant
from softlearning.samplers import rollouts

######################################################################
# NOTE: Use simulate_policy_dynamics_model.py, rather than this script
# This script is a slightly modified version of the original provided
# in the MOPO repo.
######################################################################

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment',
                        type=str,
                        help='Experiment whose policy model should be used.')
    parser.add_argument('--max-path-length', '-l', type=int, default=1000)
    parser.add_argument('--num-rollouts', '-n', type=int, default=10)
    parser.add_argument('--render-mode', '-r',
                        type=str,
                        default=None,
                        choices=('human', 'rgb_array', 'None'),
                        help="Mode to render the rollouts in.")
    parser.add_argument('--deterministic', '-d',
                        type=lambda x: bool(strtobool(x)),
                        nargs='?',
                        const=True,
                        default=True,
                        help="Evaluate policy deterministically.")

    args = parser.parse_args()
    
    if args.render_mode == 'None':
        args.render_mode = None

    return args


def simulate_policy(args):
    session = tf.keras.backend.get_session()

    exp_details = get_experiment_details(args.experiment, get_elites=False)
    experiment_path = exp_details.results_dir
    checkpoint_path = os.path.join(
        exp_details.results_dir,
        'ray_mopo',
        exp_details.environment,
        exp_details.base_dir,
        exp_details.experiment_dir,
        'checkpoint_501',
    )

    variant_path = os.path.join(experiment_path, 'params.json')
    with open(variant_path, 'r') as f:
        variant = json.load(f)

    with session.as_default():
        pickle_path = os.path.join(checkpoint_path, 'checkpoint.pkl')
        with open(pickle_path, 'rb') as f:
            picklable = pickle.load(f)

    environment_params = (
        variant['environment_params']['evaluation']
        if 'evaluation' in variant['environment_params']
        else variant['environment_params']['training'])
    evaluation_environment = get_environment_from_params(environment_params)

    policy = (
        get_policy_from_variant(variant, evaluation_environment, Qs=[None]))
    policy.set_weights(picklable['policy_weights'])

    with policy.set_deterministic(args.deterministic):
        paths = rollouts(args.num_rollouts,
                         evaluation_environment,
                         policy,
                         path_length=args.max_path_length,
                         render_mode=args.render_mode)

    #### print rewards
    rewards = [path['rewards'].sum() for path in paths]
    print('Rewards: {}'.format(rewards))
    print('Mean: {}'.format(np.mean(rewards)))
    ####
    
    # if args.render_mode != 'human':
    #     from pprint import pprint; import pdb; pdb.set_trace()
    #     pass

    return paths


if __name__ == '__main__':
    args = parse_args()
    simulate_policy(args)
