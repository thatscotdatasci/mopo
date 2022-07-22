from concurrent.futures import process
import os

import numpy as np
import matplotlib.pyplot as plt

from dogo.results import get_experiment_details
from dogo.pca.project import project_arr
from dogo.constants import ACTION_DIMS, DATA_DIR, STATE_DIMS, D4RL_NAME_DICT

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

def policy_dynamics_plot_title(dynamics_exp, policy_exp, deterministic_model, deterministic_policy):
    dynamics_exp_details = get_experiment_details(dynamics_exp)
    policy_exp_details = get_experiment_details(policy_exp)
    policy_dynamics_exp_details = get_experiment_details(policy_exp_details.dynamics_model_exp)

    if dynamics_exp_details.rex:
        dynamics_title = f'Dynamics: {dynamics_exp_details.name} - REx: True - REx Beta: {dynamics_exp_details.rex_beta} - Seed: {dynamics_exp_details.seed} - Deterministic: {deterministic_model}'
    else:
        dynamics_title = f'Dynamics: {dynamics_exp_details.name} - REx: False - Seed: {dynamics_exp_details.seed} - Deterministic: {deterministic_model}'

    if policy_dynamics_exp_details.rex:
        policy_title = f'Policy: {policy_exp_details.name} - REx: True - REx Beta: {policy_dynamics_exp_details.rex_beta} - Seed: {policy_dynamics_exp_details.seed} - Deterministic: {deterministic_policy}'
    else:
        policy_title = f'Policy: {policy_exp_details.name} - REx: False - Seed: {policy_dynamics_exp_details.seed} - Deterministic: {deterministic_policy}'
    
    return f'{dynamics_title}\n{policy_title}'

def plot_cumulative_reward(dynamics_exp, policy_exp, deterministic_model=True, deterministic_policy=True):
    fig, ax = plt.subplots(1, 1, figsize=(20,10))

    fake_unpen_rewards = retrieve_metric(dynamics_exp, policy_exp, deterministic_model, deterministic_policy, 'fake', 'unpen_rewards')
    fake_pen = retrieve_metric(dynamics_exp, policy_exp, deterministic_model, deterministic_policy, 'fake', 'reward_pens')
    eval_rewards = retrieve_metric(dynamics_exp, policy_exp, deterministic_model, deterministic_policy, 'eval', 'rewards')
    gym_rewards = retrieve_metric(dynamics_exp, policy_exp, deterministic_model, deterministic_policy, 'gym', 'rewards')

    n_records = fake_unpen_rewards.shape[1]

    for i, (metric, label) in enumerate([
        (fake_unpen_rewards, 'Learned Reward - No Penalty'),
        (eval_rewards, 'True Reward - No Penalty'),
        (fake_pen, 'MOPO Penalty'),
        (gym_rewards, 'Gym Rollout'),
    ]):
        cumsum_arr = metric.cumsum(axis=1)
        mean_arr = cumsum_arr.mean(axis=0).flatten()
        min_arr = cumsum_arr.min(axis=0).flatten()
        max_arr = cumsum_arr.max(axis=0).flatten()
        ax.plot(mean_arr, c=cols[i], label=label)
        ax.fill_between(np.arange(n_records), min_arr, max_arr, color=cols[i], alpha=0.5)

    ax.set_title(policy_dynamics_plot_title(dynamics_exp, policy_exp, deterministic_model, deterministic_policy))

    ax.legend()

def plot_reward_mse(dynamics_exp, policy_exp, deterministic_model=True, deterministic_policy=True):
    fig, ax = plt.subplots(1, 1, figsize=(20,10))

    fake_unpen_rewards = retrieve_metric(dynamics_exp, policy_exp, deterministic_model, deterministic_policy, 'fake', 'unpen_rewards')
    eval_rewards = retrieve_metric(dynamics_exp, policy_exp, deterministic_model, deterministic_policy, 'eval', 'rewards')
    squared_err = (fake_unpen_rewards-eval_rewards)**2
    mse = squared_err.mean(axis=-1).mean(axis=0)
    mse_std = squared_err.std(axis=-1).mean(axis=0)

    n_records = fake_unpen_rewards.shape[1]
    ax.plot(mse)
    ax.fill_between(np.arange(n_records), mse-2*mse_std, mse+2*mse_std, alpha=0.5)

def plot_visitation_landscape(dynamics_exp, policy_exp):
    fig, ax = plt.subplots(1, 1, figsize=(10,10))

    _, fake_pca_2d_arrs = project_arr(np.vstack(retrieve_metric(dynamics_exp, policy_exp, True, True, 'fake', 'state_action')))
    _, gym_pca_2d_arrs = project_arr(np.vstack(retrieve_metric(dynamics_exp, policy_exp, True, True, 'gym', 'state_action')))

    ax.scatter(fake_pca_2d_arrs[:,0], fake_pca_2d_arrs[:,1], marker='x', s=10, label='Model')
    ax.scatter(gym_pca_2d_arrs[:,0], gym_pca_2d_arrs[:,1], marker='x', s=10, label='Real Environment')
    ax.legend()

def plot_reward_landscape(dynamics_exp, policy_exp, deterministic_model=True, deterministic_policy=True):
    fig, ax = plt.subplots(1, 2, figsize=(20,10), subplot_kw={"projection": "3d"})

    _, fake_pca_2d_arrs = project_arr(np.vstack(retrieve_metric(dynamics_exp, policy_exp, True, True, 'fake', 'state_action')))
    _, gym_pca_2d_arrs = project_arr(np.vstack(retrieve_metric(dynamics_exp, policy_exp, True, True, 'gym', 'state_action')))
    
    fake_unpen_rewards_arrs = np.vstack(retrieve_metric(dynamics_exp, policy_exp, True, True, 'fake', 'unpen_rewards')).flatten()
    eval_rewards_arrs = np.vstack(retrieve_metric(dynamics_exp, policy_exp, True, True, 'eval', 'rewards')).flatten()
    gym_rewards_arrs = np.vstack(retrieve_metric(dynamics_exp, policy_exp, True, True, 'gym', 'rewards')).flatten()
    
    for i in range(2):
        ax[i].scatter(fake_pca_2d_arrs[:,0], fake_pca_2d_arrs[:,1], fake_unpen_rewards_arrs, marker='x', s=10, label='Learned Reward')
        ax[i].scatter(fake_pca_2d_arrs[:,0], fake_pca_2d_arrs[:,1], eval_rewards_arrs, marker='x', s=10, label='True Reward')
        ax[i].scatter(gym_pca_2d_arrs[:,0], gym_pca_2d_arrs[:,1], gym_rewards_arrs, marker='x', s=10, label='Gym Rollout')
    
    ax[0].view_init(10, 60)
    ax[1].view_init(90, 0)
    
    for i in range(2):
        ax[i].set_xlabel('PCA Dimension 1')
        ax[i].set_ylabel('PCA Dimension 2')
        ax[i].set_zlabel('Reward')
        ax[i].legend()

    fig.suptitle(policy_dynamics_plot_title(dynamics_exp, policy_exp, deterministic_model, deterministic_policy))

def plot_metric_landscape_comp(dynamics_policy_exps_list, metric):
    fig, ax = plt.subplots(1, 2, figsize=(20,10), subplot_kw={"projection": "3d"})

    for dynamics_exp, policy_exp in dynamics_policy_exps_list:
        _, fake_pca_2d_arrs = project_arr(np.vstack(retrieve_metric(dynamics_exp, policy_exp, True, True, 'fake', 'state_action')))
        fake_metric_arrs = np.vstack(retrieve_metric(dynamics_exp, policy_exp, True, True, 'fake', metric)).flatten()

        for i in range(2):
            ax[i].scatter(fake_pca_2d_arrs[:,0], fake_pca_2d_arrs[:,1], fake_metric_arrs, marker='x', s=10, label=f'{dynamics_exp} - {policy_exp}')
        
        ax[0].view_init(10, 60)
        ax[1].view_init(90, 0)

    for i in range(2):
        ax[i].set_xlabel('PCA Dimension 1')
        ax[i].set_ylabel('PCA Dimension 2')
        ax[i].set_zlabel((' '.join(metric.split('_'))).title())
        ax[i].legend()

def plot_returns_comparison(dynamics_exps, policy_exps):
    fig, ax = plt.subplots(1, 3, figsize=(40,15))
    
    dynamics_exps_labels = []
    policy_exps_labels = []
    fake_returns = np.zeros((len(dynamics_exps), len(policy_exps), 2))
    eval_returns = np.zeros((len(dynamics_exps), len(policy_exps), 2))
    gym_returns = np.zeros((len(dynamics_exps), len(policy_exps), 2))
    for i, dynamics_exp in enumerate(dynamics_exps):
        dynamics_exp_details = get_experiment_details(dynamics_exp)
        if dynamics_exp_details.rex:
            dynamics_exps_labels.append(f'{dynamics_exp} - REx: {dynamics_exp_details.rex}\nREx Beta: {dynamics_exp_details.rex_beta}\nSeed: {dynamics_exp_details.seed}')
        else:
            dynamics_exps_labels.append(f'{dynamics_exp} - REx: {dynamics_exp_details.rex}\nSeed: {dynamics_exp_details.seed}')
        
        for j, policy_exp in enumerate(policy_exps):
            if i == 0:
                policy_exp_details = get_experiment_details(policy_exp)
                policy_dynamics_exp_details = get_experiment_details(policy_exp_details.dynamics_model_exp) if policy_exp_details.dynamics_model_exp else policy_exp_details
                if policy_dynamics_exp_details.rex:
                    policy_exps_labels.append(f'{policy_dynamics_exp_details.name} - REx: {policy_dynamics_exp_details.rex}\nREx Beta: {policy_dynamics_exp_details.rex_beta}\nSeed: {dynamics_exp_details.seed}')
                else:
                    policy_exps_labels.append(f'{policy_dynamics_exp_details.name} - REx: {policy_dynamics_exp_details.rex}\nSeed: {dynamics_exp_details.seed}')

            fake_metric_arrs = retrieve_metric(dynamics_exp, policy_exp, True, True, 'fake', 'unpen_rewards').cumsum(axis=1)[:,-1,0]
            fake_returns[i,j,0], fake_returns[i,j,1] = fake_metric_arrs.mean(), fake_metric_arrs.std()

            eval_metric_arrs = retrieve_metric(dynamics_exp, policy_exp, True, True, 'eval', 'rewards').cumsum(axis=1)[:,-1,0]
            eval_returns[i,j,0], eval_returns[i,j,1] = eval_metric_arrs.mean(), eval_metric_arrs.std()

            gym_metric_arrs = retrieve_metric(dynamics_exp, policy_exp, True, True, 'gym', 'rewards').cumsum(axis=1)[:,-1,0]
            gym_returns[i,j,0], gym_returns[i,j,1] = gym_metric_arrs.mean(), gym_metric_arrs.std()
    
    for i, (res_arr, title) in enumerate([
        (fake_returns, 'Learned Dynamics'),
        (eval_returns, 'True Dynamics'),
        (gym_returns, 'Gym Rollout')
    ]):
        mat = ax[i].matshow(res_arr[:,:,0], cmap='viridis')
        ax[i].set_xticks(range(len(policy_exps)))
        ax[i].set_yticks(range(len(dynamics_exps)))
        ax[i].set_xticklabels(policy_exps_labels, rotation=45)
        ax[i].set_yticklabels(dynamics_exps_labels, rotation=45)
        ax[i].set_title(title)

        for (j,k), z in np.ndenumerate(res_arr[:,:,0]):
            if z != 0:
                ax[i].text(k, j, '{:.2f}\n±{:.2f}'.format(res_arr[j,k,0], res_arr[j,k,1]), ha="center", va="center", color='w' if z < 1000 else 'k')

    for i in range(3):
        ax[i].set_xlabel('Policy')
        ax[i].set_ylabel('Dynamics')

def get_dyanmics_pol_scores(dynamics_exps, policy_exps):
    dynamics_exps_labels = []
    policy_exps_labels = []
    fake_returns = np.zeros((len(dynamics_exps), len(policy_exps), 2))
    eval_returns = np.zeros((len(dynamics_exps), len(policy_exps), 2))
    gym_returns = np.zeros((len(dynamics_exps), len(policy_exps), 2))
    for i, dynamics_exp in enumerate(dynamics_exps):
        dynamics_exp_details = get_experiment_details(dynamics_exp)
        dynamics_exps_labels.append(f'{dynamics_exp_details.dataset}\nSeed: {dynamics_exp_details.seed}')
        
        for j, policy_exp in enumerate(policy_exps):
            if i == 0:
                policy_exp_details = get_experiment_details(policy_exp)
                policy_dynamics_exp_details = get_experiment_details(policy_exp_details.dynamics_model_exp) if policy_exp_details.dynamics_model_exp else policy_exp_details
                policy_exps_labels.append(f'{policy_dynamics_exp_details.dataset}\nSeed: {policy_dynamics_exp_details.seed}')

            fake_metric_arrs = retrieve_metric(dynamics_exp, policy_exp, True, True, 'fake', 'unpen_rewards').cumsum(axis=1)[:,-1,0]
            fake_returns[i,j,0], fake_returns[i,j,1] = fake_metric_arrs.mean(), fake_metric_arrs.std()

            eval_metric_arrs = retrieve_metric(dynamics_exp, policy_exp, True, True, 'eval', 'rewards').cumsum(axis=1)[:,-1,0]
            eval_returns[i,j,0], eval_returns[i,j,1] = eval_metric_arrs.mean(), eval_metric_arrs.std()

            gym_metric_arrs = retrieve_metric(dynamics_exp, policy_exp, True, True, 'gym', 'rewards').cumsum(axis=1)[:,-1,0]
            gym_returns[i,j,0], gym_returns[i,j,1] = gym_metric_arrs.mean(), gym_metric_arrs.std()

    return fake_returns, eval_returns, gym_returns, dynamics_exps_labels, policy_exps_labels


def plot_returns_comparison_pol_dep(dynamics_exps, policy_exps):
    fig, ax = plt.subplots(1, 3, figsize=(33,10))
    
    fake_returns, eval_returns, gym_returns, dynamics_exps_labels, policy_exps_labels = get_dyanmics_pol_scores(dynamics_exps, policy_exps)
    
    for i, (res_arr, title) in enumerate([
        (fake_returns, 'Learned Dynamics'),
        (eval_returns, 'True Dynamics'),
        (gym_returns, 'Gym Rollout')
    ]):
        mat = ax[i].matshow(res_arr[:,:,0], cmap='viridis')
        ax[i].set_xticks(range(len(policy_exps)))
        ax[i].set_yticks(range(len(dynamics_exps)))
        ax[i].set_xticklabels(policy_exps_labels, rotation=45)

        if title != 'Gym Rollout':
            ax[i].set_yticklabels(dynamics_exps_labels, rotation=45)
        else:
            ax[i].set_yticklabels([])
        ax[i].set_title(title)

        for (j,k), z in np.ndenumerate(res_arr[:,:,0]):
            if z != 0:
                ax[i].text(k, j, '{:.2f}\n±{:.2f}'.format(res_arr[j,k,0], res_arr[j,k,1]), ha="center", va="center", color='w' if z < 1000 else 'k')

    for i in range(3):
        ax[i].set_xlabel('Policy')
        ax[i].set_ylabel('Dynamics')

def plot_pen_reward_landscape_comp(dynamics_policy_exps_list, pen_coeff=1.0):
    fig, ax = plt.subplots(1, 2, figsize=(20,10), subplot_kw={"projection": "3d"})

    for dynamics_exp, policy_exp in dynamics_policy_exps_list:
        _, fake_pca_2d_arrs = project_arr(np.vstack(retrieve_metric(dynamics_exp, policy_exp, True, True, 'fake', 'state_action')))
        
        fake_unpen_rewards_arrs = np.vstack(retrieve_metric(dynamics_exp, policy_exp, True, True, 'fake', 'unpen_rewards')).flatten()
        fake_reward_pens_arrs = np.vstack(retrieve_metric(dynamics_exp, policy_exp, True, True, 'fake', 'reward_pens')).flatten()

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

def plot_reward_error_landscape_comp(dynamics_policy_exps_list, training_dataset=False):
    fig, ax = plt.subplots(1, 2, figsize=(20,10), subplot_kw={"projection": "3d"})

    for dynamics_exp, policy_exp in dynamics_policy_exps_list:
        _, fake_pca_2d_arrs = project_arr(np.vstack(retrieve_metric(dynamics_exp, policy_exp, True, True, 'fake', 'state_action')))

        fake_unpen_rewards_arrs = np.vstack(retrieve_metric(dynamics_exp, policy_exp, True, True, 'fake', 'unpen_rewards')).flatten()
        eval_rewards_arrs = np.vstack(retrieve_metric(dynamics_exp, policy_exp, True, True, 'eval', 'rewards')).flatten()
        squared_err_arrs = (fake_unpen_rewards_arrs-eval_rewards_arrs)**2

        for i in range(2):
            ax[i].scatter(fake_pca_2d_arrs[:,0], fake_pca_2d_arrs[:,1], squared_err_arrs, marker='x', s=10, label=f'{dynamics_exp} - {policy_exp}')
        
        ax[0].view_init(10, 60)
        ax[1].view_init(90, 0)
    
    if training_dataset:
        dynamics_exp_details = get_experiment_details(dynamics_exp)
        _, data_pca_2d_arr = project_arr(np.load(os.path.join(DATA_DIR, f'{dynamics_exp_details.dataset}.npy'))[:,:STATE_DIMS+ACTION_DIMS])
        for i in range(2):
            ax[i].scatter(data_pca_2d_arr[:,0], data_pca_2d_arr[:,1], -0.1*np.ones_like(data_pca_2d_arr[:,1]), marker='+', s=10, color='lightgray', label=f'Training Data')

    for i in range(2):
        ax[i].set_xlabel('PCA Dimension 1')
        ax[i].set_ylabel('PCA Dimension 2')
        ax[i].set_zlabel('Reward Squared Error')
        ax[i].legend()
