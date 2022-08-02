from concurrent.futures import process
from genericpath import isfile
import os

import numpy as np
import matplotlib.pyplot as plt

from dogo.results import get_experiment_details, PoolArrs
from dogo.pca.project import project_arr, learn_project_arr_2d
from dogo.constants import ACTION_DIMS, DATA_DIR, STATE_DIMS, RESULTS_BASEDIR
from dogo.visualisation.model_pool_plotting import (
    model_pool_pen_rewards_2dhist,
    model_pool_penalties_2dhist,
    model_pool_rmse_2dhist,
    model_pool_unpen_rewards_2dhist,
    model_pool_visitation_2dhist
)

class MetricNotFound(Exception):
    pass

cols = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def retrieve_metric(dynamics_exp, policy_exp, deterministic_model, deterministic_policy, env, metric):
    metric_path = os.path.join(RESULTS_BASEDIR, 'analysis', 'policy', f'{dynamics_exp}_{policy_exp}_dm{deterministic_model}_dp{deterministic_policy}_{env}_{metric}.npy')
    if os.path.isfile(metric_path):
        return np.load(metric_path)
    else:
        print(f'Could not find: {metric_path}')
        raise MetricNotFound()

def retrieve_metric_stats(dynamics_exp, policy_exp, deterministic_model, deterministic_policy, env, metric):
    arr = retrieve_metric(dynamics_exp, policy_exp, deterministic_model, deterministic_policy, env, metric)
    return {
        'mean': np.nanmean(arr, axis=-1).flatten(),
        'min': np.nanmin(arr, axis=-1).flatten(),
        'max': np.nanmax(arr, axis=-1).flatten(),
        'std': np.nanstd(arr, axis=-1).flatten()
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

def plot_cumulative_reward(policy_exp, dynamics_exp=None, deterministic_model=True, deterministic_policy=True):
    fig, ax = plt.subplots(1, 1, figsize=(20,10))

    if dynamics_exp is None:
        dynamics_exp = get_experiment_details(policy_exp).dynamics_model_exp

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

def plot_reward_mse(policy_exp, dynamics_exp=None, deterministic_model=True, deterministic_policy=True):
    fig, ax = plt.subplots(1, 1, figsize=(20,10))

    if dynamics_exp is None:
        dynamics_exp = get_experiment_details(policy_exp).dynamics_model_exp

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

def policy_dynamics_pool_visitation_pca(policy_exp_list_label_set):
    policy_exp_lists = [i[0] for i in policy_exp_list_label_set]
    policy_exps = [i for j in policy_exp_lists for i in j]

    sa_arrs = []
    for policy_exp in policy_exps:
        policy_exp_details = get_experiment_details(policy_exp)
        dynamics_exp = policy_exp_details.dynamics_model_exp

        try:
            fake_sa_arrs = np.vstack(retrieve_metric(dynamics_exp, policy_exp, True, True, 'fake', 'state_action'))
            gym_sa_arrs = np.vstack(retrieve_metric(dynamics_exp, policy_exp, True, True, 'gym', 'state_action'))
            sa_arrs.append(np.vstack((fake_sa_arrs, gym_sa_arrs)))
        except MetricNotFound:
            continue
    
    sa_arrs = np.vstack(sa_arrs)

    pca_2d = learn_project_arr_2d(sa_arrs)
    _, _, explained_var = project_arr(sa_arrs, pca_2d=pca_2d)

    try:
        assert np.round(pca_2d.explained_variance_ratio_.sum(), 8) == np.round(explained_var, 8)
    except AssertionError as e:
        print(pca_2d.explained_variance_ratio_.sum(), explained_var)
        raise e

    return learn_project_arr_2d(np.vstack(sa_arrs))

def policy_dynamics_pool_visitation_2dhist(policy_exp_list_label_set, mode, eval_env, vmin=None, vmax=None, pca_model=None):
    results_arr = {}
    policy_exp_lists = [i[0] for i in policy_exp_list_label_set]
    policy_exps = [i for j in policy_exp_lists for i in j]
    for policy_exp in policy_exps:
        policy_exp_details = get_experiment_details(policy_exp)
        dynamics_exp = policy_exp_details.dynamics_model_exp

        try:
            if eval_env == 'fake':
                _, pca_2d_arrs, explained_var = project_arr(np.vstack(retrieve_metric(dynamics_exp, policy_exp, True, True, 'fake', 'state_action')), pca_2d=pca_model)
                n_records = len(pca_2d_arrs)
                fake_unpen_rewards_arrs = np.vstack(retrieve_metric(dynamics_exp, policy_exp, True, True, 'fake', 'unpen_rewards'))
                fake_reward_pens_arrs = np.vstack(retrieve_metric(dynamics_exp, policy_exp, True, True, 'fake', 'reward_pens'))
                eval_rewards_arrs = np.vstack(retrieve_metric(dynamics_exp, policy_exp, True, True, 'eval', 'rewards'))
                
                reward_arrs = fake_unpen_rewards_arrs - fake_reward_pens_arrs * policy_exp_details.penalty_coeff
                reward_pens_arrs = fake_reward_pens_arrs
                mse_arrs = (fake_unpen_rewards_arrs-eval_rewards_arrs)**2
            elif eval_env == 'gym':
                _, pca_2d_arrs, explained_var = project_arr(np.vstack(retrieve_metric(dynamics_exp, policy_exp, True, True, 'gym', 'state_action')), pca_2d=pca_model)
                n_records = len(pca_2d_arrs)
                reward_arrs = np.vstack(retrieve_metric(dynamics_exp, policy_exp, True, True, 'gym', 'rewards'))
                reward_pens_arrs = np.zeros((n_records, 1))
                mse_arrs = np.zeros((n_records, 1))
            else:
                raise RuntimeError(f'Do not recognise evaluation environment: {eval_env}')
            
            pool = np.hstack((
                np.zeros((n_records, STATE_DIMS+ACTION_DIMS+STATE_DIMS)), reward_arrs, np.zeros((n_records, 2)), reward_pens_arrs
            ))
            results_arr[policy_exp] = PoolArrs(pool=pool, pca_sa_1d=None, pca_sa_2d=pca_2d_arrs, mse_results=mse_arrs, explained_var_2d=explained_var)
        except MetricNotFound:
            results_arr[policy_exp] = None
    
    if mode=='visitation':
        model_pool_visitation_2dhist(policy_exp_list_label_set, vmin=vmin, vmax=vmax, results_arr=results_arr)
    elif mode=='pen-rewards':
        model_pool_pen_rewards_2dhist(policy_exp_list_label_set, vmin=vmin, vmax=vmax, results_arr=results_arr)
    elif mode=='unpen-rewards':
        model_pool_unpen_rewards_2dhist(policy_exp_list_label_set, pen_coeff=policy_exp_details.penalty_coeff, vmin=vmin, vmax=vmax, results_arr=results_arr)
    elif mode=='penalties':
        model_pool_penalties_2dhist(policy_exp_list_label_set, vmin=vmin, vmax=vmax, results_arr=results_arr)
    elif mode=='rmse':
        model_pool_rmse_2dhist(policy_exp_list_label_set, vmin=vmin, vmax=vmax, results_arr=results_arr)
    else:
        raise RuntimeError(f'Cannot handle mode: {mode}')

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

def get_policy_values(policy_exps, expected_records, deterministic_model=True, deterministic_policy=True):
    xtick_labels = []
    fake_pen_returns_arr = []
    fake_unpen_returns_arr = []
    eval_returns_arr = []
    gym_returns_arr = []
    for i, policy_exp in enumerate(policy_exps):
        policy_exp_details = get_experiment_details(policy_exp)
        dynamics_exp = policy_exp_details.dynamics_model_exp
        mopo_pen_coeff = policy_exp_details.penalty_coeff

        if policy_exp_details.rex:
            exp_label = f'REx $\\beta={policy_exp_details.rex_beta}$\nSeed: {policy_exp_details.seed}'
        else:
            exp_label = f'No REx\nSeed: {policy_exp_details.seed}'
        xtick_labels.append(exp_label)

        try:
            fake_unpen_rewards = retrieve_metric(dynamics_exp, policy_exp, deterministic_model, deterministic_policy, 'fake', 'unpen_rewards')
            fake_pen = retrieve_metric(dynamics_exp, policy_exp, deterministic_model, deterministic_policy, 'fake', 'reward_pens')
            fake_pen_rewards = fake_unpen_rewards - fake_pen*mopo_pen_coeff
            
            eval_rewards = retrieve_metric(dynamics_exp, policy_exp, deterministic_model, deterministic_policy, 'eval', 'rewards')
            gym_rewards = retrieve_metric(dynamics_exp, policy_exp, deterministic_model, deterministic_policy, 'gym', 'rewards')
        except MetricNotFound:
            print(f'Metric not found for: {policy_exp}')
            fake_pen_returns_arr.append(np.ones(expected_records)*np.nan)
            fake_unpen_returns_arr.append(np.ones(expected_records)*np.nan)
            eval_returns_arr.append(np.ones(expected_records)*np.nan)
            gym_returns_arr.append(np.ones(expected_records)*np.nan)
        else:
            fake_pen_returns_arr.append(fake_pen_rewards.sum(axis=1).flatten())
            fake_unpen_returns_arr.append(fake_unpen_rewards.sum(axis=1).flatten())
            eval_returns_arr.append(eval_rewards.sum(axis=1).flatten())
            gym_returns_arr.append(gym_rewards.sum(axis=1).flatten())

    fake_pen_returns_arr = np.vstack(fake_pen_returns_arr)
    fake_unpen_returns_arr = np.vstack(fake_unpen_returns_arr)
    eval_returns_arr = np.vstack(eval_returns_arr)
    gym_returns_arr = np.vstack(gym_returns_arr)

    return fake_pen_returns_arr, fake_unpen_returns_arr, eval_returns_arr, gym_returns_arr, xtick_labels

def plot_policy_values(policy_exps, expected_records, deterministic_model=True, deterministic_policy=True, show_penalised=True):
    fig, ax = plt.subplots(1, 1, figsize=(20,10))

    fake_pen_returns_arr, fake_unpen_returns_arr, eval_returns_arr, gym_returns_arr, xtick_labels = get_policy_values(policy_exps, expected_records, deterministic_model, deterministic_policy)
    fake_pen_returns_mean, fake_pen_returns_std = np.nanmean(fake_pen_returns_arr, axis=1), np.nanstd(fake_pen_returns_arr, axis=1)
    fake_unpen_returns_mean, fake_unpen_returns_std = np.nanmean(fake_unpen_returns_arr, axis=1), np.nanstd(fake_unpen_returns_arr, axis=1)
    eval_returns_mean, eval_returns_std = np.nanmean(eval_returns_arr, axis=1), np.nanstd(eval_returns_arr, axis=1)
    gym_returns_mean, gym_returns_std = np.nanmean(gym_returns_arr, axis=1), np.nanstd(gym_returns_arr, axis=1)

    if show_penalised:
        ax.errorbar(np.arange(len(policy_exps)), fake_pen_returns_mean, fake_pen_returns_std, ls='', marker='x', label='Learned Returns - Penalised')
    ax.errorbar(np.arange(len(policy_exps)), fake_unpen_returns_mean, fake_unpen_returns_std, ls='', marker='x', label='Learned Returns - Unpenalised')
    ax.errorbar(np.arange(len(policy_exps)), eval_returns_mean, eval_returns_std, ls='', marker='x', label='Evaluation Returns')
    ax.errorbar(np.arange(len(policy_exps)), gym_returns_mean, gym_returns_std, ls='', marker='x', label='Gym Returns')

    # ax.set_title(policy_dynamics_plot_title(dynamics_exp, policy_exp, deterministic_model, deterministic_policy))
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Average Return')
    ax.set_xticks(np.arange(len(policy_exps)))
    ax.set_xticklabels(xtick_labels)

    ax.legend()

def plot_policy_group_values(policy_exp_label_groups, expected_records, deterministic_model=True, deterministic_policy=True, show_penalised=True):
    fig, ax = plt.subplots(1, 1, figsize=(20,10))

    xtick_labels = []
    grouped_fake_pen_returns = np.zeros((len(policy_exp_label_groups),2))
    grouped_fake_unpen_returns = np.zeros((len(policy_exp_label_groups),2))
    grouped_eval_returns = np.zeros((len(policy_exp_label_groups),2))
    grouped_gym_returns = np.zeros((len(policy_exp_label_groups),2))
    for i, (policy_exps, label) in enumerate(policy_exp_label_groups):
        xtick_labels.append(label)
        fake_pen_returns_arr, fake_unpen_returns_arr, eval_returns_arr, gym_returns_arr, _ = get_policy_values(policy_exps, expected_records, deterministic_model, deterministic_policy)
        grouped_fake_pen_returns[i,0], grouped_fake_pen_returns[i,1] = np.nanmean(fake_pen_returns_arr), np.nanstd(fake_pen_returns_arr)
        grouped_fake_unpen_returns[i,0], grouped_fake_unpen_returns[i,1] = np.nanmean(fake_unpen_returns_arr), np.nanstd(fake_unpen_returns_arr)
        grouped_eval_returns[i,0], grouped_eval_returns[i,1] = np.nanmean(eval_returns_arr), np.nanstd(eval_returns_arr)
        grouped_gym_returns[i,0], grouped_gym_returns[i,1] = np.nanmean(gym_returns_arr), np.nanstd(gym_returns_arr)

    if show_penalised:
        ax.errorbar(np.arange(len(policy_exp_label_groups)), grouped_fake_pen_returns[:,0], grouped_fake_pen_returns[:,1], ls='', marker='x', label='Learned Dynamics - Penalised')
    ax.errorbar(np.arange(len(policy_exp_label_groups)), grouped_fake_unpen_returns[:,0], grouped_fake_unpen_returns[:,1], ls='', marker='x', label='Learned Dynamics - Unpenalised')
    ax.errorbar(np.arange(len(policy_exp_label_groups)), grouped_eval_returns[:,0], grouped_eval_returns[:,1], ls='', marker='x', label='Evaluation Environment')
    ax.errorbar(np.arange(len(policy_exp_label_groups)), grouped_gym_returns[:,0], grouped_gym_returns[:,1], ls='', marker='x', label='Real Environment')

    # ax.set_title(policy_dynamics_plot_title(dynamics_exp, policy_exp, deterministic_model, deterministic_policy))
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Average Return')
    ax.set_xticks(np.arange(len(policy_exp_label_groups)))
    ax.set_xticklabels(xtick_labels)

    ax.legend()

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
            fake_metric_arrs = fake_metric_arrs[np.abs(fake_metric_arrs)<=50000]
            fake_returns[i,j,0], fake_returns[i,j,1] = np.nanmean(fake_metric_arrs), np.nanstd(fake_metric_arrs)

            eval_metric_arrs = retrieve_metric(dynamics_exp, policy_exp, True, True, 'eval', 'rewards').cumsum(axis=1)[:,-1,0]
            eval_metric_arrs = eval_metric_arrs[np.abs(eval_metric_arrs)<=50000]
            eval_returns[i,j,0], eval_returns[i,j,1] = np.nanmean(eval_metric_arrs), np.nanstd(eval_metric_arrs)

            gym_metric_arrs = retrieve_metric(dynamics_exp, policy_exp, True, True, 'gym', 'rewards').cumsum(axis=1)[:,-1,0]
            gym_returns[i,j,0], gym_returns[i,j,1] = np.nanmean(gym_metric_arrs), np.nanstd(gym_metric_arrs)
    
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
            fake_metric_arrs = fake_metric_arrs[np.abs(fake_metric_arrs)<=50000]
            fake_returns[i,j,0], fake_returns[i,j,1] = np.nanmean(fake_metric_arrs), np.nanstd(fake_metric_arrs)

            eval_metric_arrs = retrieve_metric(dynamics_exp, policy_exp, True, True, 'eval', 'rewards').cumsum(axis=1)[:,-1,0]
            eval_metric_arrs = eval_metric_arrs[np.abs(eval_metric_arrs)<=50000]
            eval_returns[i,j,0], eval_returns[i,j,1] = np.nanmean(eval_metric_arrs), np.nanstd(eval_metric_arrs)

            gym_metric_arrs = retrieve_metric(dynamics_exp, policy_exp, True, True, 'gym', 'rewards').cumsum(axis=1)[:,-1,0]
            gym_returns[i,j,0], gym_returns[i,j,1] = np.nanmean(gym_metric_arrs), np.nanstd(gym_metric_arrs)

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

def plot_returns_comparison_pol_dep_groups(dynamics_exp_groups_labels, policy_exp_groups_labels):
    fig, ax = plt.subplots(1, 3, figsize=(33,10))
    
    n_dynamics_groups = len(dynamics_exp_groups_labels)
    n_policy_groups = len(policy_exp_groups_labels)
    fake_returns = np.zeros((n_dynamics_groups, n_policy_groups, 2))
    eval_returns = np.zeros((n_dynamics_groups, n_policy_groups, 2))
    gym_returns = np.zeros((n_dynamics_groups, n_policy_groups, 2))
    for i, (dynamic_exp_group, _) in enumerate(dynamics_exp_groups_labels):
        for j, (policy_exp_group, _) in enumerate(policy_exp_groups_labels):
            group_fake_returns, group_eval_returns, group_gym_returns, _, _ = get_dyanmics_pol_scores(dynamic_exp_group, policy_exp_group)
            fake_returns[i,j,0], fake_returns[i,j,1] = np.nanmean(group_fake_returns[:,:,0]), np.nanmean(group_fake_returns[:,:,1])
            eval_returns[i,j,0], eval_returns[i,j,1] = np.nanmean(group_eval_returns[:,:,0]), np.nanmean(group_eval_returns[:,:,1])
            gym_returns[i,j,0], gym_returns[i,j,1]   = np.nanmean(group_gym_returns[:,:,0]), np.nanmean(group_gym_returns[:,:,1])
    
    for i, (res_arr, title) in enumerate([
        (fake_returns, 'Learned Dynamics'),
        (eval_returns, 'True Dynamics'),
        (gym_returns, 'Gym Rollout')
    ]):
        mat = ax[i].matshow(res_arr[:,:,0], cmap='viridis')
        ax[i].set_xticks(range(n_policy_groups))
        ax[i].set_yticks(range(n_dynamics_groups))
        ax[i].set_xticklabels([i[1] for i in policy_exp_groups_labels], rotation=45)

        if title != 'Gym Rollout':
            ax[i].set_yticklabels([i[1] for i in dynamics_exp_groups_labels], rotation=45)
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
