from ray import tune
import numpy as np
import pdb

from softlearning.misc.utils import get_git_rev, deep_update

M = 256
REPARAMETERIZE = True

NUM_COUPLING_LAYERS = 2

GAUSSIAN_POLICY_PARAMS_BASE = {
    'type': 'GaussianPolicy',
    'kwargs': {
        'hidden_layer_sizes': (M, M),
        'squash': True,
    }
}

GAUSSIAN_POLICY_PARAMS_FOR_DOMAIN = {}

POLICY_PARAMS_BASE = {
    'GaussianPolicy': GAUSSIAN_POLICY_PARAMS_BASE,
}

POLICY_PARAMS_BASE.update({
    'gaussian': POLICY_PARAMS_BASE['GaussianPolicy'],
})

POLICY_PARAMS_FOR_DOMAIN = {
    'GaussianPolicy': GAUSSIAN_POLICY_PARAMS_FOR_DOMAIN,
}

POLICY_PARAMS_FOR_DOMAIN.update({
    'gaussian': POLICY_PARAMS_FOR_DOMAIN['GaussianPolicy'],
})

DEFAULT_MAX_PATH_LENGTH = 1000
MAX_PATH_LENGTH_PER_DOMAIN = {
    'Point2DEnv': 50,
    'Point2DWallEnv': 50,
    'Pendulum': 200,
}

ALGORITHM_PARAMS_ADDITIONAL = {
    'MOPO': {
        'type': 'MOPO',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'store_extra_policy_info': False,
            'action_prior': 'uniform',
            'n_initial_exploration_steps': int(5000),
        }
    },
    'SQL': {
        'type': 'SQL',
        'kwargs': {
            'policy_lr': 3e-4,
            'target_update_interval': 1,
            'n_initial_exploration_steps': int(1e3),
            'reward_scale': tune.sample_from(lambda spec: (
                {
                    'Swimmer': 30,
                    'Hopper': 30,
                    'HalfCheetah': 30,
                    'HalfCheetahJump': 30,
                    'Walker2d': 10,
                    'Ant': 300,
                    'AntAngle': 300,
                    'Humanoid': 100,
                    'Pendulum': 1,
                }.get(
                    spec.get('config', spec)
                    ['environment_params']
                    ['training']
                    ['domain'],
                    1.0
                ),
            )),
        }
    },
    'MVE': {
        'type': 'MVE',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            'store_extra_policy_info': False,
            'action_prior': 'uniform',
            'n_initial_exploration_steps': int(5000),
        }
    },
}

DEFAULT_NUM_EPOCHS = 200

NUM_EPOCHS_PER_DOMAIN = {
    'Swimmer': int(3e3),
    'Hopper': int(1e3),
    'HalfCheetah': int(500),#int(3e3),
    'HalfCheetahJump': int(3e3),
    'HalfCheetahVel': int(500),#int(3e3),
    'HalfCheetahVelJump': int(3e3),
    'Walker2d': int(3e3),
    'Ant': int(1000),#int(500),#int(3e3),
    'AntAngle': int(3e3),
    'Humanoid': int(1e4),
    'Pusher2d': int(2e3),
    'HandManipulatePen': int(1e4),
    'HandManipulateEgg': int(1e4),
    'HandManipulateBlock': int(1e4),
    'HandReach': int(1e4),
    'Point2DEnv': int(100),
    'Point2DWallEnv': int(100),
    'Reacher': int(200),
    'Pendulum': 10,
}

ALGORITHM_PARAMS_PER_DOMAIN = {
    **{
        domain: {
            'kwargs': {
                'n_epochs': NUM_EPOCHS_PER_DOMAIN.get(
                    domain, DEFAULT_NUM_EPOCHS),
                'n_initial_exploration_steps': (
                    MAX_PATH_LENGTH_PER_DOMAIN.get(
                        domain, DEFAULT_MAX_PATH_LENGTH
                    ) * 10),
            }
        } for domain in NUM_EPOCHS_PER_DOMAIN
    }
}

ENVIRONMENT_PARAMS = {
    'Swimmer': {  # 2 DoF
    },
    'Hopper': {  # 3 DoF
    },
    'HalfCheetah': {  # 6 DoF
    },
    'HalfCheetahJump': {  # 6 DoF
    },
    'HalfCheetahVel': {  # 6 DoF
    },
    'HalfCheetahVelJump': {  # 6 DoF
    },
    'Walker2d': {  # 6 DoF
    },
    'Ant': {  # 8 DoF
        'Parameterizable-v3': {
            'healthy_reward': 0.0,
            'healthy_z_range': (-np.inf, np.inf),
            'exclude_current_positions_from_observation': False,
        }
    },
    'AntAngle': {  # 8 DoF
        'Parameterizable-v3': {
            'healthy_reward': 0.0,
            'healthy_z_range': (-np.inf, np.inf),
            'exclude_current_positions_from_observation': False,
        }
    },
    'Humanoid': {  # 17 DoF
        'Parameterizable-v3': {
            'healthy_reward': 0.0,
            'healthy_z_range': (-np.inf, np.inf),
            'exclude_current_positions_from_observation': False,
        }
    },
    'Pusher2d': {  # 3 DoF
        'Default-v3': {
            'arm_object_distance_cost_coeff': 0.0,
            'goal_object_distance_cost_coeff': 1.0,
            'goal': (0, -1),
        },
        'DefaultReach-v0': {
            'arm_goal_distance_cost_coeff': 1.0,
            'arm_object_distance_cost_coeff': 0.0,
        },
        'ImageDefault-v0': {
            'image_shape': (32, 32, 3),
            'arm_object_distance_cost_coeff': 0.0,
            'goal_object_distance_cost_coeff': 3.0,
        },
        'ImageReach-v0': {
            'image_shape': (32, 32, 3),
            'arm_goal_distance_cost_coeff': 1.0,
            'arm_object_distance_cost_coeff': 0.0,
        },
        'BlindReach-v0': {
            'image_shape': (32, 32, 3),
            'arm_goal_distance_cost_coeff': 1.0,
            'arm_object_distance_cost_coeff': 0.0,
        }
    },
    'Point2DEnv': {
        'Default-v0': {
            'observation_keys': ('observation', 'desired_goal'),
        },
        'Wall-v0': {
            'observation_keys': ('observation', 'desired_goal'),
        },
        'Offline-v0': {
            'observation_keys': ('observation', 'desired_goal'),
        },
    },
    'Point2DWallEnv': {
        'Offline-v0': {
            'observation_keys': ('observation', 'desired_goal'),
        },
    }
}

NUM_CHECKPOINTS = 10


def get_variant_spec_base(universe, domain, task, policy, algorithm, env_params):
    algorithm_params = deep_update(
        env_params,
        ALGORITHM_PARAMS_PER_DOMAIN.get(domain, {})
    )
    algorithm_params = deep_update(
        algorithm_params,
        ALGORITHM_PARAMS_ADDITIONAL.get(algorithm, {})
    )
    variant_spec = {
        # 'git_sha': get_git_rev(),

        'environment_params': {
            'training': {
                'domain': domain,
                'task': task,
                'universe': universe,
                'kwargs': (
                    ENVIRONMENT_PARAMS.get(domain, {}).get(task, {})),
            },
            'evaluation': tune.sample_from(lambda spec: (
                spec.get('config', spec)
                ['environment_params']
                ['training']
            )),
        },
        'policy_params': deep_update(
            POLICY_PARAMS_BASE[policy],
            POLICY_PARAMS_FOR_DOMAIN[policy].get(domain, {})
        ),
        'Q_params': {
            'type': 'double_feedforward_Q_function',
            'kwargs': {
                'hidden_layer_sizes': (M, M),
            }
        },
        'algorithm_params': algorithm_params,
        'replay_pool_params': {
            'type': 'SimpleReplayPool',
            'kwargs': {
                'max_size': tune.sample_from(lambda spec: (
                    {
                        'SimpleReplayPool': int(1e6),
                        'TrajectoryReplayPool': int(1e4),
                    }.get(
                        spec.get('config', spec)
                        ['replay_pool_params']
                        ['type'],
                        int(1e6))
                )),
            }
        },
        'sampler_params': {
            'type': 'SimpleSampler',
            'kwargs': {
                'max_path_length': MAX_PATH_LENGTH_PER_DOMAIN.get(
                    domain, DEFAULT_MAX_PATH_LENGTH),
                'min_pool_size': MAX_PATH_LENGTH_PER_DOMAIN.get(
                    domain, DEFAULT_MAX_PATH_LENGTH),
                'batch_size': 256,
            }
        },
        'run_params': {
            'seed': env_params.seed or tune.sample_from(lambda spec: np.random.randint(0, 10000)),
            'checkpoint_at_end': True,
            'checkpoint_frequency': NUM_EPOCHS_PER_DOMAIN.get(
                domain, DEFAULT_NUM_EPOCHS) // NUM_CHECKPOINTS,
            'checkpoint_replay_pool': False,
        },
    }

    return variant_spec

def get_variant_spec(args, env_params):
    universe, domain, task = env_params.universe, env_params.domain, env_params.task
    variant_spec = get_variant_spec_base(
        universe, domain, task, args.policy, env_params.type, env_params)

    if args.checkpoint_replay_pool is not None:
        variant_spec['run_params']['checkpoint_replay_pool'] = (
            args.checkpoint_replay_pool)
    return variant_spec
