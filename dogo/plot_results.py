import os
from itertools import zip_longest

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import mark_inset
from scipy.ndimage import uniform_filter1d
from sklearn.linear_model import LinearRegression

from dogo.constants import FIG_DIR
from dogo.results import get_scores_df, get_results, get_experiment_details

lss = ['-', '--']
cols = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 'b', 'g', 'r']

FEATURE_LABEL_DICT = {
    'evaluation/return-average': 'Evaluation Return Average',
    'Q_loss': 'Q Loss',
    'Q-avg': 'Average Q-Value',
    'model/mean_unpenalized_rewards': 'Average Unpenalised Rewards',
    'model/mean_penalized_rewards': 'Average Penalised Reward',
    'model/mean_penalty': 'Average Penalty'
}

SCORING_LABEL_DICT = {
    'log_prob': 'Log Likelihood',
    'overall_mse': 'MSE'
}

def plot_experiment_metrics(
    metric_collection: list, exp_set_label_collection: list, fig_shape: tuple, starting_epoch=50, 
    running_avg_size=None, x_label='Epochs', fig_size=None, loc='upper left', show_terminals=True, 
    save_path=None, terminals_size=100, ins_loc=None, ins_range=None, legend_font_size=18, epoch_limit=None
):
    fig_size = fig_size or (fig_shape[1]*10,fig_shape[0]*10)
    fig, ax = plt.subplots(*fig_shape, figsize=fig_size)

    plt.rcParams.update({'font.size': 18})

    if fig_shape[0] > 1 and fig_shape[1] > 1:
        get_ax =lambda i: ax[i//fig_shape[1], i%fig_shape[1]]
    elif fig_shape[1] > 1:
        get_ax =lambda i: ax[i]
    else:
        get_ax = lambda i: ax

    if ins_loc:
        ins_x_min, ins_x_max, ins_y_min, ins_y_max = ins_range
        ins = get_ax(None).inset_axes(ins_loc)

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

                get_ax(i).plot(x_vals[:epoch_limit], mean_arr[:epoch_limit], c=cols[k], ls=lss[j], label=legend_label)
                get_ax(i).fill_between(x_vals[:epoch_limit], min_arr[:epoch_limit], max_arr[:epoch_limit], color=cols[k], alpha=0.25)

                if show_terminals:
                    terminal_points = np.sort(np.argmax(np.isnan(comb_arr), axis=0))[1:]
                    get_ax(i).scatter(x_vals[terminal_points], mean_arr[terminal_points], color='r', s=terminals_size, zorder=1000+i)

                if ins_range:
                    ins.plot(x_vals, mean_arr, c=cols[k], ls=lss[j], label=legend_label)
                    ins.fill_between(x_vals, min_arr, max_arr, color=cols[k], alpha=0.25)
                    ins.set_xlim(ins_x_min, ins_x_max)
                    ins.set_ylim(ins_y_min, ins_y_max)

        get_ax(i).set_xlabel(x_label)
        get_ax(i).set_ylabel(y_label)
        get_ax(i).set_ylim(y_lim)
        get_ax(i).legend(loc=loc, prop={'size': legend_font_size})

    if ins_loc:
        mark_inset(get_ax(None), ins, loc1=1, loc2=3, fc="none", lw=2, ec='r')

    if save_path is not None:
        fig.savefig(os.path.join(FIG_DIR, save_path), pad_inches=0.2, bbox_inches='tight')


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

            print(f'{label}: {np.mean(vals_arr)}')

        ax.errorbar(labels, mean_vals, np.abs(np.vstack((min_vals, max_vals))-mean_vals), label=f'{"Maximum" if i==0 else "Minumum"} Penalty')

    ax.set_xlabel('Experiment Set')
    ax.set_ylabel('Penalty')
    ax.legend()

def plot_evaluation_returns(exps: list, title: str = None, xmin=None, xmax=None, ymin=None, ymax=None, feature: str = 'evaluation/return-average', fig_size=(20,10), labels=None, legend=True, save_path=None, erm_dashed=True, loc='upper left', ncol=1):
    plt.rc('font', size=12)
    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    for i, exp in enumerate(exps):
        if labels is not None:
            label = labels[i]
        else:
            label = f'{exp.name} - {exp.dataset} - REx: {exp.rex} - Beta: {exp.rex_beta} - Seed: {exp.seed}'

        if erm_dashed:
            ls='--' if not exp.rex else '-'
        else:
            ls='-'

        ax.plot(
            exp.sac.result['timesteps_total'], exp.sac.result[feature], c=cols[i], ls=ls, label=label
        )
        print(f'{exp.name}: {exp.sac.result[feature].values[-1]}')

        # if feature == 'model/mean_penalty':
        #     ax.axhline(exp.min_penalty, ls=':', c=cols[i])
        #     ax.axhline(exp.max_penalty, ls=':', c=cols[i])

    ax.set_xlabel('Steps')
    ax.set_ylabel(FEATURE_LABEL_DICT[feature])
    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(bottom=ymin, top=ymax)
    
    if title is not None:
        ax.set_title(title)
    
    if legend:
        ax.legend(loc=loc, ncol=ncol)

    if save_path is not None:
        fig.savefig(os.path.join(FIG_DIR, save_path), pad_inches=0.2, bbox_inches='tight')

def plot_grouped_evaluation_returns(exp_set_labels: list, title: str = None, xmin=-1000, ymin=None, ymax=None, show_ends=True, feature: str = 'evaluation/return-average', save_path=None, loc='upper left', max_timestep=500000):
    fig, ax = plt.subplots(1, 1, figsize=(18,10))
    plt.rcParams.update({'font.size': 18})

    summary_metrics = {}
    for i, (exp_set, label) in enumerate(exp_set_labels):
        exp_results = []
        exp_end_points = []

        for exp in exp_set:
            exp_timesteps = exp.sac.result['timesteps_total'].values
            exp_timesteps = exp_timesteps[exp_timesteps<=max_timestep]

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
        count_arr = np.sum(~np.isnan(comb_arr), axis=-1)
        x_vals = np.arange(len(mean_arr))*1000

        ax.plot(x_vals, mean_arr, c=cols[i], label=label)
        # ax.fill_between(x_vals, min_arr, max_arr, color=cols[i], alpha=0.5)
        ax.fill_between(x_vals, mean_arr-std_arr, mean_arr+std_arr, color=cols[i], alpha=0.5)

        terminal_points = np.where(np.sort((comb_arr==np.NaN).argmin(axis=0))>0)[0]
        ax.scatter(x_vals[terminal_points], mean_arr[terminal_points], color=cols[i], s=100)

        if show_ends:
            exp_end_points = exp_end_points[exp_end_points<x_vals[-1]]
            ax.scatter(exp_end_points, -2000*np.ones_like(exp_end_points), marker='x', color=cols[i], s=100)
        
        summary_metrics[label] = {
            'mean': np.round(mean_arr[-1],0).astype(int),
            'std': np.round(std_arr[-1],0).astype(int),
            'count': count_arr[-2],
            'text': f'{np.round(mean_arr[-1],0).astype(int)} ± {np.round(std_arr[-1],0).astype(int)}'
        }
        

    ax.set_xlabel('Training Steps')
    ax.set_ylabel(FEATURE_LABEL_DICT[feature])
    ax.set_xlim(left=xmin, right=max_timestep+1000)
    ax.set_ylim(bottom=ymin, top=ymax)
    if title is not None:
        ax.set_title(title)
    ax.legend(loc=loc)

    if save_path is not None:
        fig.savefig(os.path.join(FIG_DIR, save_path), pad_inches=0.2, bbox_inches='tight')

    return summary_metrics

def plot_dynamics_score_vs_agent_return(exp_groups, eval_ds, train_eval_ds, agent_score_timesteps=500000, xmin=None, xmax=None, ymin1=None, ymax1=None, primary_metric='log_prob', secondary_metric='overall_mse', show_secondary_metric=True, mode='ood'):
    fig, ax1 = plt.subplots(1, 1, figsize=(10,10))

    if show_secondary_metric:
        ax2 = ax1.twinx()

    scores_all = []
    primary_metric_vals_all = []
    for i, (exps, label) in enumerate(exp_groups):
        primary_metric_vals = []
        secondary_metric_vals = []
        scores = []
        for exp_name in exps:
            exp_results = get_results(exp_name)
            terminus_mask = exp_results.sac.result['timesteps_total']==agent_score_timesteps
            if any(terminus_mask):
                exp = get_experiment_details(exp_name)
                dynamics_exp = get_experiment_details(exp.dynamics_model_exp)
                dynamics_scores = get_scores_df([dynamics_exp.name], eval_ds)

                if mode == 'ood':
                    primary_metric_vals.append(dynamics_scores.loc[~dynamics_scores['evaluation_dataset'].isin(train_eval_ds), primary_metric].mean())
                    secondary_metric_vals.append(dynamics_scores.loc[~dynamics_scores['evaluation_dataset'].isin(train_eval_ds), secondary_metric].mean())
                elif mode == 'id':
                    primary_metric_vals.append(dynamics_scores.loc[dynamics_scores['evaluation_dataset'].isin(train_eval_ds), primary_metric].mean())
                    secondary_metric_vals.append(dynamics_scores.loc[dynamics_scores['evaluation_dataset'].isin(train_eval_ds), secondary_metric].mean())
                elif mode == 'all':
                    primary_metric_vals.append(dynamics_scores[primary_metric].mean())
                    secondary_metric_vals.append(dynamics_scores[secondary_metric].mean())
                scores.append(exp_results.sac.result.loc[terminus_mask, 'evaluation/return-average'].values[0])

        scores_all.extend(scores)
        primary_metric_vals_all.extend(primary_metric_vals)
        
        arg_sort = np.argsort(scores)
        scores = np.array(scores)[arg_sort]
        primary_metric_vals = np.array(primary_metric_vals)[arg_sort]
        secondary_metric_vals = np.array(secondary_metric_vals)[arg_sort]

        if show_secondary_metric:
            ax1.plot(scores, primary_metric_vals, marker='o', ls='--', color=cols[i], label=f'{label} - Log-Likelihood')
            ax2.plot(scores, secondary_metric_vals, marker='x', ls='--', color=cols[i], label=f'{label} - MSE')
        else:
            ax1.plot(primary_metric_vals, scores, marker='o', ls='--', color=cols[i], label=f'{label}')
    
    if show_secondary_metric:
        ax1.set_xlabel('Agent Return')
        ax1.set_ylabel(SCORING_LABEL_DICT[primary_metric])
        ax2.set_ylabel(SCORING_LABEL_DICT[secondary_metric])
    else:
        arg_sort_all = np.argsort(scores_all)
        scores_all = np.array(scores_all)[arg_sort_all]
        primary_metric_vals_all = np.array(primary_metric_vals_all)[arg_sort_all]

        lr = LinearRegression().fit(primary_metric_vals_all.reshape(-1, 1), scores_all.reshape(-1, 1))
        x_range = np.linspace(primary_metric_vals_all.min(), primary_metric_vals_all.max(), 2)
        y_vals = lr.predict(x_range.reshape(-1, 1)).flatten()

        ax1.plot(x_range, y_vals, marker='', ls=':', color=cols[i+1], label=f'LBF - $R^2={lr.score(primary_metric_vals_all.reshape(-1, 1), scores_all.reshape(-1, 1)):.2f}$')

        ax1.set_ylabel('Agent Return')
        ax1.set_xlabel(SCORING_LABEL_DICT[primary_metric])

    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin1, ymax1)

    fig.legend(ncol=2, loc='upper center')
