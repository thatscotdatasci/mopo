import os

import numpy as np
import matplotlib.pyplot as plt

cols = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def retrieve_metric(dynamics_exp, policy_exp, deterministic_model, deterministic_policy, env, metric):
    return np.load(os.path.join('policy', f'{dynamics_exp}_{policy_exp}_dm{deterministic_model}_dp{deterministic_policy}_{env}_{metric}.npy'))

def retrieve_metric_stats(dynamics_exp, policy_exp, deterministic_model, deterministic_policy, env, metric):
    arr = retrieve_metric(dynamics_exp, policy_exp, deterministic_model, deterministic_policy, env, metric)
    return {
        'mean': arr.mean(axis=-1).flatten(),
        'min': arr.min(axis=-1).flatten(),
        'max': arr.max(axis=-1).flatten(),
        'std': arr.std(axis=-1).flatten()
    }

def plot_cumulative_reward(dynamics_exp, policy_exp):
    fig, ax = plt.subplots(1, 1, figsize=(20,10))

    fake_unpen_rewards = retrieve_metric(dynamics_exp, policy_exp, True, True, 'fake', 'unpen_rewards')
    fake_pen = retrieve_metric(dynamics_exp, policy_exp, True, True, 'fake', 'reward_pens')
    eval_rewards = retrieve_metric(dynamics_exp, policy_exp, True, True, 'eval', 'rewards')
    gym_rewards = retrieve_metric(dynamics_exp, policy_exp, True, True, 'gym', 'rewards')

    n_records = len(fake_unpen_rewards)

    for i, (metric, label) in enumerate([
        (fake_unpen_rewards, 'Model'),
        (fake_pen, 'Penalty'),
        (eval_rewards, 'Eval'),
        (gym_rewards, 'Gym'),
    ]):
        cumsum_arr = metric.cumsum(axis=0)
        mean_arr = cumsum_arr.mean(axis=-1).flatten()
        min_arr = cumsum_arr.min(axis=-1).flatten()
        max_arr = cumsum_arr.max(axis=-1).flatten()
        ax.plot(mean_arr, c=cols[i], label=label)
        ax.fill_between(np.arange(n_records), min_arr, max_arr, color=cols[i], alpha=0.5)

    ax.legend()

def plot_visitation_landscape(dynamics_exp, policy_exp):
    fig, ax = plt.subplots(1, 1, figsize=(10,10))

    fake_pca_2d_arrs = retrieve_metric(dynamics_exp, policy_exp, True, True, 'fake', 'state_action_pca_2d').transpose(2,0,1).flatten()
    gym_pca_2d_arrs = retrieve_metric(dynamics_exp, policy_exp, True, True, 'gym', 'state_action_pca_2d').transpose(2,0,1).flatten()

    ax.scatter(fake_pca_2d_arrs[:,0], fake_pca_2d_arrs[:,1], marker='x', s=10, label='Model')
    ax.scatter(gym_pca_2d_arrs[:,0], gym_pca_2d_arrs[:,1], marker='x', s=10, label='Real Environment')
    ax.legend()

def plot_reward_landscape(dynamics_exp, policy_exp):
    fig, ax = plt.subplots(1, 2, figsize=(20,10), subplot_kw={"projection": "3d"})

    fake_pca_2d_arrs = retrieve_metric(dynamics_exp, policy_exp, True, True, 'fake', 'state_action_pca_2d').swapaxes(1,2).reshape(-1,2)
    gym_pca_2d_arrs = retrieve_metric(dynamics_exp, policy_exp, True, True, 'gym', 'state_action_pca_2d').swapaxes(1,2).reshape(-1,2)
    
    fake_unpen_rewards_arrs = retrieve_metric(dynamics_exp, policy_exp, True, True, 'fake', 'unpen_rewards').transpose(2,0,1).flatten()
    eval_rewards_arrs = retrieve_metric(dynamics_exp, policy_exp, True, True, 'eval', 'rewards').transpose(2,0,1).flatten()
    gym_rewards_arrs = retrieve_metric(dynamics_exp, policy_exp, True, True, 'gym', 'rewards').transpose(2,0,1).flatten()
    
    for i in range(2):
        ax[i].scatter(fake_pca_2d_arrs[:,0], fake_pca_2d_arrs[:,1], fake_unpen_rewards_arrs, marker='x', s=10, label='Model')
        ax[i].scatter(fake_pca_2d_arrs[:,0], fake_pca_2d_arrs[:,1], eval_rewards_arrs, marker='x', s=10, label='Eval')
        ax[i].scatter(gym_pca_2d_arrs[:,0], gym_pca_2d_arrs[:,1], gym_rewards_arrs, marker='x', s=10, label='Gym')
    
    ax[0].view_init(10, 60)
    ax[1].view_init(90, 0)
    
    for i in range(2):
        ax[i].set_xlabel('PCA Dimension 1')
        ax[i].set_ylabel('PCA Dimension 2')
        ax[i].set_zlabel('Reward')
        ax[i].legend()

def plot_metric_landscape_comp(dynamics_policy_exps_list, metric):
    fig, ax = plt.subplots(1, 2, figsize=(20,10), subplot_kw={"projection": "3d"})

    for dynamics_exp, policy_exp in dynamics_policy_exps_list:
        fake_pca_2d_arrs = retrieve_metric(dynamics_exp, policy_exp, True, True, 'fake', 'state_action_pca_2d').transpose(2,0,1).reshape(-1,2)
        fake_metric_arrs = retrieve_metric(dynamics_exp, policy_exp, True, True, 'fake', metric).transpose(2,0,1).flatten()

        for i in range(2):
            ax[i].scatter(fake_pca_2d_arrs[:,0], fake_pca_2d_arrs[:,1], fake_metric_arrs, marker='x', s=10, label=f'{dynamics_exp} - {policy_exp}')
        
        ax[0].view_init(10, 60)
        ax[1].view_init(90, 0)

    for i in range(2):
        ax[i].set_xlabel('PCA Dimension 1')
        ax[i].set_ylabel('PCA Dimension 2')
        ax[i].set_zlabel((' '.join(metric.split('_'))).title())
        ax[i].legend()

def plot_pen_reward_landscape_comp(dynamics_policy_exps_list, pen_coeff=1.0):
    fig, ax = plt.subplots(1, 2, figsize=(20,10), subplot_kw={"projection": "3d"})

    for dynamics_exp, policy_exp in dynamics_policy_exps_list:
        fake_pca_2d_arrs = retrieve_metric(dynamics_exp, policy_exp, True, True, 'fake', 'state_action_pca_2d').transpose(2,0,1).reshape(-1,2)
        
        fake_unpen_rewards_arrs = retrieve_metric(dynamics_exp, policy_exp, True, True, 'fake', 'unpen_rewards').transpose(2,0,1).flatten()
        fake_reward_pens_arrs = retrieve_metric(dynamics_exp, policy_exp, True, True, 'fake', 'reward_pens').transpose(2,0,1).flatten()

        fake_pen_reward_arrs = fake_unpen_rewards_arrs - (float(pen_coeff) * fake_reward_pens_arrs)

        for i in range(2):
            ax[i].scatter(fake_pca_2d_arrs[:,0], fake_pca_2d_arrs[:,1], fake_pen_reward_arrs, marker='x', s=10, label=f'{dynamics_exp} - {policy_exp}')
        
        ax[0].view_init(10, 60)
        ax[1].view_init(90, 0)

    for i in range(2):
        ax[i].set_xlabel('PCA Dimension 1')
        ax[i].set_ylabel('PCA Dimension 2')
        ax[i].set_zlabel('Pen Reward')
        ax[i].legend()
