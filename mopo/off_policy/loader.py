import os
import glob
import pickle
import gzip
import pdb
import numpy as np

def restore_pool(replay_pool, experiment_root, max_size, save_path=None, policy_type='default'):
    print("'d4rl' in experiment_root", 'd4rl' in experiment_root)
    if 'd4rl' in experiment_root:
        print('experiment_root[5:]', experiment_root[5:])
        restore_pool_d4rl(replay_pool, experiment_root[5:])
    else:
        assert os.path.exists(experiment_root)
        if os.path.isdir(experiment_root):
            print('path exists')
            print('max_size', max_size)
            restore_pool_softlearning(replay_pool, experiment_root, max_size, save_path)
        else:
            try:
                restore_pool_contiguous(replay_pool, experiment_root, policy_type=policy_type)
            except Exception as e:
                # Do not suppress an exception from restore_pool_contiguous
                # Expecting that this method should work, and will not be using bear
                raise e
                # restore_pool_bear(replay_pool, experiment_root)
    print('[ mbpo/off_policy ] Replay pool has size: {}'.format(replay_pool.size))


def restore_pool_d4rl(replay_pool, name):
    import gym
    import d4rl
    data = d4rl.qlearning_dataset(gym.make(name))
    data['rewards'] = np.expand_dims(data['rewards'], axis=1)
    data['terminals'] = np.expand_dims(data['terminals'], axis=1)

    # Treat all data as having come from the same policy
    data['policies'] = np.zeros_like(data['rewards'])
    data['penalties'] = np.zeros_like(data['rewards'])

    replay_pool.add_samples(data)


def restore_pool_softlearning(replay_pool, experiment_root, max_size, save_path=None):
    print('[ mopo/off_policy ] Loading SAC replay pool from: {}'.format(experiment_root))
    experience_paths = [
        checkpoint_dir
        for checkpoint_dir in sorted(glob.iglob(
            os.path.join(experiment_root, 'checkpoint_*')))
    ]

    checkpoint_epochs = [int(path.split('_')[-1]) for path in experience_paths]
    checkpoint_epochs = sorted(checkpoint_epochs)
    if max_size == 250e3:
        checkpoint_epochs = checkpoint_epochs[2:]

    for epoch in checkpoint_epochs:
        fullpath = os.path.join(experiment_root, 'checkpoint_{}'.format(epoch), 'replay_pool.pkl')
        print('[ mopo/off_policy ] Loading replay pool data: {}'.format(fullpath))
        replay_pool.load_experience(fullpath)
        if replay_pool.size >= max_size:
            break

    if save_path is not None:
        size = replay_pool.size
        stat_path = os.path.join(save_path, 'pool_stat_{}.pkl'.format(size))
        save_path = os.path.join(save_path, 'pool_{}.pkl'.format(size))
        d = {}
        for key in replay_pool.fields.keys():
            d[key] = replay_pool.fields[key][:size]

        num_paths = 0
        temp = 0
        path_end_idx = []
        for i in range(d['terminals'].shape[0]):
            if d['terminals'][i] or i - temp + 1 == 1000:
                num_paths += 1
                temp = i + 1
                path_end_idx.append(i)
        total_return = d['rewards'].sum()
        avg_return = total_return / num_paths
        buffer_max, buffer_min = -np.inf, np.inf
        path_return = 0.0
        for i in range(d['rewards'].shape[0]):
            path_return += d['rewards'][i]
            if i in path_end_idx:
                if path_return > buffer_max:
                    buffer_max = path_return
                if path_return < buffer_min:
                    buffer_min = path_return
                path_return = 0.0

        print('[ mopo/off_policy ] Replay pool average return is {}, buffer_max is {}, buffer_min is {}'.format(avg_return, buffer_max, buffer_min))
        d_stat = dict(avg_return=avg_return, buffer_max=buffer_max, buffer_min=buffer_min)
        pickle.dump(d_stat, open(stat_path, 'wb'))

        print('[ mopo/off_policy ] Saving replay pool to: {}'.format(save_path))
        pickle.dump(d, open(save_path, 'wb'))


def restore_pool_bear(replay_pool, load_path):
    print('[ mopo/off_policy ] Loading BEAR replay pool from: {}'.format(load_path))
    data = pickle.load(gzip.open(load_path, 'rb'))
    num_trajectories = data['terminals'].sum() or 1000
    avg_return = data['rewards'].sum() / num_trajectories
    print('[ mopo/off_policy ] {} trajectories | avg return: {}'.format(num_trajectories, avg_return))

    for key in ['log_pis', 'data_policy_mean', 'data_policy_logvar']:
        del data[key]

    replay_pool.add_samples(data)


def restore_pool_contiguous(replay_pool, load_path):
    print('[ mopo/off_policy ] Loading contiguous replay pool from: {}'.format(load_path))
    import numpy as np
    data = np.load(load_path)

    state_dim = replay_pool.fields['observations'].shape[1]
    action_dim = replay_pool.fields['actions'].shape[1]

    # +1 for rewards, +1 for terminals, +1 for policy indicator
    expected_dim = state_dim + action_dim + state_dim + 1 + 1 + 1
    actual_dim = data.shape[1]

    if actual_dim == expected_dim:
        print('[ mopo/off_policy ] Pool includes policy identifier')
    elif actual_dim == expected_dim - 1:
        print('[ mopo/off_policy ] Pool does not include policy identifier - adding')
        policy_id = np.full((data.shape[0], 1), 0)
        data = np.hstack((data, policy_id))
    else:
        assert False, 'Expected {} dimensions (inc. optional policy identifier), found {}'.format(expected_dim, actual_dim)

    dims = [state_dim, action_dim, state_dim, 1, 1, 1]
    ends = []
    current_end = 0
    for d in dims:
        current_end += d
        ends.append(current_end)
    states, actions, next_states, rewards, dones, policies = np.split(data, ends, axis=1)[:6]

    def get_limits(arr, N=5):
        size = len(arr) // N
        limits = sorted(arr)[::size][1:N]
        return limits

    if policy_type == 'reward_partioned':
        limits = get_limits(rewards[:, 0], N=5)
        rewards_policy = (rewards > limits).sum(-1)
        policies = rewards_policy

    if policy_type in ['trajectory', 'value_partitioned']:
        trajectories = np.zeros_like(dones, dtype=int)
        cur_trajectory_index = 0
        for i in range(len(dones)):
            if dones[i]:
                cur_trajectory_index += 1
            trajectories[i] = cur_trajectory_index

        if policy_type == 'trajectory_partitioned':
            policies = trajectories
        if policy_type == 'value_partitioned':
            one_hot_trajectories = np.eye(trajectories.max() + 1)[trajectories[:, 0]]
            sum_rewards_per_tragectories = one_hot_trajectories.T @ rewards
            n_transactions_per_tragectories = one_hot_trajectories.T.sum(-1)
            value_transactions_per_tragectories = sum_rewards_per_tragectories[:, 0] / n_transactions_per_tragectories
            values = value_transactions_per_tragectories[trajectories]
            limits = get_limits(values[:, 0])
            value_policy = (values > limits).sum(-1)
            policies = value_policy

    print(f'number of samples in each {policy_type} partition:', [(policies == i).sum() for i in range(policies.max() + 1)])

    replay_pool.add_samples({
        'observations': states,
        'actions': actions,
        'next_observations': next_states,
        'rewards': rewards,
        'terminals': dones.astype(bool),
        'policies': policies,
        'penalties': np.zeros_like(rewards)
    })
