from itertools import zip_longest

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

lss = ['-', '--']
cols = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


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

                terminal_points = np.sort(comb_arr.argmin(axis=0))[:-1]
                get_ax(i).scatter(x_vals[terminal_points], mean_arr[terminal_points], color=cols[k], s=100)

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

def plot_grouped_evaluation_returns(exp_set_steps: list):
    fig, ax = plt.subplots(1, 1, figsize=(20,10))

    summary_metrics = {}
    feature = 'evaluation/return-average'
    for i, (exp_set, steps) in enumerate(exp_set_steps):
        exp_results = []
        exp_break_points = []

        for exp in exp_set:
            latest_start = np.where(exp.sac.result['timesteps_total'].values==1000)[0][-1]
            break_points = np.where(exp.sac.result['timesteps_total'].values[1:]==1000)[0]

            exp_results.append(exp.sac.result[feature][latest_start:])
            exp_break_points.append(break_points)
        
        comb_arr = np.vstack(list(zip_longest(*[
            exp_res for exp_res in exp_results
            ], fillvalue=np.NaN
        )))
        exp_break_points = np.hstack(exp_break_points)

        mean_arr = np.nanmean(comb_arr, axis=-1)
        min_arr = np.nanmin(comb_arr, axis=-1)
        max_arr = np.nanmax(comb_arr, axis=-1)
        std_arr = np.nanstd(comb_arr, axis=-1)
        x_vals = np.arange(len(mean_arr))*100

        ax.plot(x_vals, mean_arr, c=cols[i], label=f'Beta: {steps}M')
        ax.fill_between(x_vals, min_arr, max_arr, color=cols[i], alpha=0.5)

        terminal_points = np.where(np.sort((comb_arr==np.NaN).argmin(axis=0))>0)[0]
        ax.scatter(x_vals[terminal_points], mean_arr[terminal_points], color=cols[i], s=100)

        ax.scatter(x_vals[exp_break_points], -2000*np.ones_like(exp_break_points), marker='x', color=cols[i], s=100)

        summary_metrics[steps] = {
            'mean': mean_arr[-1],
            'std': std_arr[-1]
        }
        

    ax.set_xlabel('Steps')
    ax.set_ylabel('Evaluation Return Average')
    # ax.set_xlim(-1000,501000)
    # ax.set_ylim(-3000,12500)
    ax.legend(loc='upper left')

    return summary_metrics
