import os
from itertools import zip_longest

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

from dogo.constants import FIG_DIR

lss = ['-', '--']
cols = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

FEATURE_LABEL_DICT = {
    'evaluation/return-average': 'Evaluation Return Average',
    'Q_loss': 'Q Loss',
    'Q-avg': 'Average Q-Value',
    'model/mean_unpenalized_rewards': 'Treaning Mean Unpenalised Reward'
}

def plot_experiment_metrics(metric_collection: list, exp_set_label_collection: list, fig_shape: tuple, starting_epoch=50, running_avg_size=None, x_label='Epochs'):
    fig, ax = plt.subplots(*fig_shape, figsize=(fig_shape[1]*10,fig_shape[0]*10))

    if fig_shape[0] > 1 and fig_shape[1] > 1:
        get_ax =lambda i: ax[i//fig_shape[1], i%fig_shape[1]]
    elif fig_shape[1] > 1:
        get_ax =lambda i: ax[i]
    else:
        get_ax = lambda i: ax

    for i, (metric, y_label, y_lim) in enumerate(metric_collection):
        for j, exp_set_lables in enumerate(exp_set_label_collection):
            for k, (exp_set, legend_label) in enumerate(exp_set_lables):
                comb_arr= np.vstack(list(zip_longest(*[
                    getattr(exp.dynamics, metric).mean(axis=1).values for exp in exp_set
                    ],
                    fillvalue=np.NaN
                )))[starting_epoch:]

                mean_arr = np.nanmean(comb_arr, axis=-1)
                min_arr = np.nanmin(comb_arr, axis=-1)
                max_arr = np.nanmax(comb_arr, axis=-1)
                x_vals = np.arange(len(mean_arr)) + starting_epoch

                if running_avg_size:
                    mean_arr = uniform_filter1d(mean_arr, size=running_avg_size, mode='reflect')
                    min_arr = uniform_filter1d(min_arr, size=running_avg_size, mode='reflect')
                    max_arr = uniform_filter1d(max_arr, size=running_avg_size, mode='reflect')

                get_ax(i).plot(x_vals, mean_arr, c=cols[k], ls=lss[j], label=legend_label)
                get_ax(i).fill_between(x_vals, min_arr, max_arr, color=cols[k], alpha=0.25)

                terminal_points = np.sort(np.argmax(np.isnan(comb_arr), axis=0))[1:]
                get_ax(i).scatter(x_vals[terminal_points], mean_arr[terminal_points], color='r', s=100)

        get_ax(i).set_xlabel(x_label)
        get_ax(i).set_ylabel(y_label)
        get_ax(i).set_ylim(y_lim)
        get_ax(i).legend(loc='upper left')


def plot_min_max_logvars(exp_set_label_collection: list):
    fig, ax = plt.subplots(1, 1, figsize=(20,10))

    for i, (exp_set, legend_label) in enumerate(exp_set_label_collection):
        for j, metric in enumerate(['max_logvars', 'min_logvars']):
            vals_arr= np.vstack([
                getattr(exp, metric).flatten() for exp in exp_set
            ]).T

            mean_arr = np.mean(vals_arr, axis=-1)
            min_arr = np.min(vals_arr, axis=-1)
            max_arr = np.max(vals_arr, axis=-1)
            x_vals = np.arange(len(mean_arr))

            ax.plot(x_vals, mean_arr, c=cols[i], ls=lss[j], label=f'{legend_label} - {"Max" if j==0 else "Min"}')
            ax.fill_between(x_vals, min_arr, max_arr, color=cols[i], alpha=0.25)

    ax.set_xlabel('Feature')
    ax.set_ylabel('Log Variance')
    ax.set_xticks(x_vals)
    ax.legend()


def plot_min_max_penalty(exp_set_label_collection: list):
    fig, ax = plt.subplots(1, 1, figsize=(20,10))

    for i, metric in enumerate(['max_penalty', 'min_penalty']):
        mean_vals, min_vals, max_vals, labels = [], [], [], []
        for j, (exp_set, label) in enumerate(exp_set_label_collection):
            vals_arr= [getattr(exp, metric)for exp in exp_set]

            mean_vals.append(np.mean(vals_arr))
            min_vals.append(np.min(vals_arr))
            max_vals.append(np.max(vals_arr))
            labels.append(label)

        ax.errorbar(labels, mean_vals, np.abs(np.vstack((min_vals, max_vals))-mean_vals), label=f'{"Maximum" if i==0 else "Minumum"} Penalty')

    ax.set_xlabel('Experiment Set')
    ax.set_ylabel('Penalty')
    ax.legend()

def plot_evaluation_returns(exps: list, title: str = None, ymin=None, ymax=None, feature: str = 'evaluation/return-average'):
    fig, ax = plt.subplots(1, 1, figsize=(20,10))

    for exp in exps:
        ax.plot(
            exp.sac.result['timesteps_total'], exp.sac.result[feature], ls='--' if not exp.rex else '-', label=f'{exp.name} - {exp.dataset} - REx: {exp.rex} - Beta: {exp.rex_beta} - Seed: {exp.seed}'
        )
    ax.set_xlabel('Steps')
    ax.set_ylabel(FEATURE_LABEL_DICT[feature])
    ax.set_ylim(bottom=ymin, top=ymax)
    if title is not None:
        ax.set_title(title)
    ax.legend()

def plot_grouped_evaluation_returns(exp_set_labels: list, title: str = None, xmin=-1000, xmax=501000, ymin=None, ymax=None, show_ends=True, feature: str = 'evaluation/return-average', save_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(18,10))
    plt.rcParams.update({'font.size': 18})

    summary_metrics = {}
    for i, (exp_set, label) in enumerate(exp_set_labels):
        exp_results = []
        exp_end_points = []

        for exp in exp_set:
            exp_timesteps = exp.sac.result['timesteps_total'].values
            start_points = np.hstack([np.where(exp_timesteps==1000)[0], len(exp_timesteps)])
            exp_end_points.append(exp_timesteps[start_points[1:]-1])

            if len(start_points) > 1:
                training_lengths = np.ediff1d(start_points)
                training_start_ind = np.argmax(training_lengths)
                training_start = start_points[training_start_ind]
                training_end = len(exp_timesteps) if training_start_ind == len(start_points) else start_points[training_start_ind+1]

                exp_results.append(exp.sac.result[feature][training_start:training_end])
        
        comb_arr = np.vstack(list(zip_longest(*[
            exp_res for exp_res in exp_results
            ], fillvalue=np.NaN
        )))
        exp_end_points = np.hstack(exp_end_points)

        mean_arr = np.nanmean(comb_arr, axis=-1)
        min_arr = np.nanmin(comb_arr, axis=-1)
        max_arr = np.nanmax(comb_arr, axis=-1)
        std_arr = np.nanstd(comb_arr, axis=-1)
        x_vals = np.arange(len(mean_arr))*1000

        ax.plot(x_vals, mean_arr, c=cols[i], label=label)
        ax.fill_between(x_vals, min_arr, max_arr, color=cols[i], alpha=0.5)

        terminal_points = np.where(np.sort((comb_arr==np.NaN).argmin(axis=0))>0)[0]
        ax.scatter(x_vals[terminal_points], mean_arr[terminal_points], color=cols[i], s=100)

        if show_ends:
            exp_end_points = exp_end_points[exp_end_points<x_vals[-1]]
            ax.scatter(exp_end_points, -2000*np.ones_like(exp_end_points), marker='x', color=cols[i], s=100)
        
        summary_metrics[label] = {
            'mean': np.round(mean_arr[-1],0).astype(int),
            'std': np.round(std_arr[-1],0).astype(int)
        }
        

    ax.set_xlabel('Training Steps')
    ax.set_ylabel(FEATURE_LABEL_DICT[feature])
    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(bottom=ymin, top=ymax)
    if title is not None:
        ax.set_title(title)
    ax.legend(loc='upper left')

    if save_path is not None:
        fig.savefig(os.path.join(FIG_DIR, save_path), pad_inches=0.2, bbox_inches='tight')

    return summary_metrics
