import os

import numpy as np
import matplotlib.pyplot as plt

from dogo.results import get_experiment_details
from dogo.pca.project import project_arr

cols = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def retrieve_metric(metric, exp, dataset):
    return np.load(os.path.join('dynamics', f'{exp}_{dataset}_{metric}.npy'))

def plot_model_log_norms(exp_name, dataset):
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    x_vals, _ = project_arr(retrieve_metric('state_action', exp_name, dataset))

    exp = get_experiment_details(exp_name)

    vals_arr = retrieve_metric('ensemble_stds_norm', exp.name, dataset)
    for i, y_vals in enumerate(vals_arr.T):
        ax.scatter(x_vals, y_vals, marker='x', color=cols[i], label=f'Model: {i}')
    ax.axhline(exp.max_penalty, ls='--', color='k')
    ax.axhline(exp.min_penalty, ls='-.', color='k')

    ax.set_xlabel('PCA Dimension 1')
    ax.set_ylabel('Frobenious Norm')
    ax.set_title(f'{exp.name} - REx: {exp.rex} - Beta: {exp.rex_beta} - Seed: {exp.seed}')
    ax.legend()

def plot_metric_over_ranges(experiment_names, metric, dataset, loglog=True, lower_xlim=None, upper_xlim=None, lower_ylim=None, upper_ylim=None):
    fig, ax = plt.subplots(1, 1, figsize=(10,10))

    for i, exp_name in enumerate(experiment_names):
        x_vals, _ = project_arr(retrieve_metric('state_action', exp_name, dataset))

        exp = get_experiment_details(exp_name)
        vals_arr = retrieve_metric(metric, exp.name, dataset)
        
        x_vals = vals_arr.mean(axis=1)
        x_vals_sorted = np.sort(x_vals)

        y_vals = np.cumsum(np.ones_like(x_vals_sorted))/len(x_vals_sorted)

        plot_func = ax.loglog if loglog else ax.plot
        plot_func(x_vals_sorted, y_vals, color=cols[i], label=f'{exp.name} - REx: {exp.rex} - Beta: {exp.rex_beta} - Seed: {exp.seed}')

        if metric == 'reward_pens':
            ax.axhline(exp.max_penalty, ls='--', color=cols[i])

    ax.set_xlabel(' '.join(metric.split('_')).title())
    ax.set_ylabel('Cumulative Fraction of Records')
    ax.set_xlim(left=lower_xlim, right=upper_xlim)
    ax.set_ylim(bottom=lower_ylim, top=upper_ylim)
    ax.legend()

def plot_metric_2d(experiment_names, metric, dataset, lower_ylim=None, upper_ylim=None):
    fig, ax = plt.subplots(1, 1, figsize=(10,10))

    for i, exp_name in enumerate(experiment_names):
        x_vals, _ = project_arr(retrieve_metric('state_action', exp_name, dataset))

        exp = get_experiment_details(exp_name)
        vals_arr = retrieve_metric(metric, exp.name, dataset)
        y_vals = vals_arr.mean(axis=1)

        if lower_ylim or upper_ylim:
            per_in_lims = ((y_vals >= (lower_ylim or -np.inf)) & (y_vals <= (upper_ylim or np.inf))).mean() * 100
        else:
            per_in_lims = 100

        ax.scatter(x_vals, y_vals, marker='x', color=cols[i], label=f'{exp.name} - REx: {exp.rex} - Beta: {exp.rex_beta} - Seed: {exp.seed} - Visible: {per_in_lims:.2f} %')

        if metric == 'reward_pens':
            ax.axhline(exp.max_penalty, ls='--', color=cols[i])

    ax.set_xlabel('PCA Dimension 1')
    ax.set_ylabel(' '.join(metric.split('_')).title())
    ax.set_ylim(bottom=lower_ylim, top=upper_ylim)
    ax.legend()

def plot_metric_3d(experiment_names, metric, dataset):
    fig, ax = plt.subplots(1, 2, figsize=(20,10), subplot_kw={"projection": "3d"})

    for exp_name in experiment_names:
        _, x_vals = project_arr(retrieve_metric('state_action', exp_name, dataset))

        exp = get_experiment_details(exp_name)
        vals_arr = retrieve_metric(metric, exp.name, dataset)
        y_vals = vals_arr.mean(axis=1)

        for i in range(2):
            ax[i].scatter(x_vals[:,0], x_vals[:,1], y_vals, marker='x', label=f'{exp.name} - REx: {exp.rex} - Beta: {exp.rex_beta} - Seed: {exp.seed}')

        ax[0].view_init(10, 60)
        ax[1].view_init(10, -60)
    
    for i in range(2):
        ax[i].set_xlabel('PCA Dimension 1')
        ax[i].set_ylabel('PCA Dimension 2')
        ax[i].set_zlabel(metric)
        ax[i].legend()
