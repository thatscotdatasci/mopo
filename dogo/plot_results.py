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
    'model/eval_return_mean': 'Dynamics Return Average',
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
    save_path=None, terminals_size=100, ins_loc=None, ins_range=None, legend_font_size=18, epoch_limit=None,
    font_size=18
):
    """ Function for plotting any dynamics model training metrics. The large number of arguments allow significant
    control over what's displayed, including constraining the axes and plotting running averages.
    """

    fig_size = fig_size or (fig_shape[1]*10,fig_shape[0]*10)
    fig, ax = plt.subplots(*fig_shape, figsize=fig_size)

    plt.rcParams.update({'font.size': font_size})

    # Helper function for correctly indexing the subplots, depending on the number of rows and columns
    if fig_shape[0] > 1 and fig_shape[1] > 1:
        get_ax =lambda i: ax[i//fig_shape[1], i%fig_shape[1]]
    elif fig_shape[1] > 1:
        get_ax =lambda i: ax[i]
    else:
        get_ax = lambda _: ax

    # Create an inset plot
    if ins_loc:
        ins_x_min, ins_x_max, ins_y_min, ins_y_max = ins_range
        ins = get_ax(None).inset_axes(ins_loc)

    # Loop over the metric to be plotted
    for i, (metric, y_label, y_lim) in enumerate(metric_collection):
        # Loop over the experiment collections to be plotted
        for j, exp_set_lables in enumerate(exp_set_label_collection):
            # Loop over the experiment sets, whose results are to be averaged
            for k, (exp_set, legend_label) in enumerate(exp_set_lables):

                # The early stopping criterion introduces variation in training times.
                # For those expeirments which ran for less time, fill the end of the results arrays with np.NaN
                # such that all arrays are the same length and can be stacked.
                comb_arr= np.vstack(list(zip_longest(*[
                    getattr(exp.dynamics, metric).mean(axis=1).values for exp in exp_set
                    ],
                    fillvalue=np.NaN
                )))[starting_epoch:]

                # Determine the mean, minimum and maximum values at each step of training
                # Ignore the np.NaN values that were added in the previous step
                mean_arr = np.nanmean(comb_arr, axis=-1)
                min_arr = np.nanmin(comb_arr, axis=-1)
                max_arr = np.nanmax(comb_arr, axis=-1)
                x_vals = np.arange(len(mean_arr)) + starting_epoch

                # Determine the running averages, if the `running_avg_size` parameter is specified
                if running_avg_size:
                    mean_arr = uniform_filter1d(mean_arr, size=running_avg_size, mode='reflect')
                    min_arr = uniform_filter1d(min_arr, size=running_avg_size, mode='reflect')
                    max_arr = uniform_filter1d(max_arr, size=running_avg_size, mode='reflect')

                # Plot the mean
                get_ax(i).plot(x_vals[:epoch_limit], mean_arr[:epoch_limit], c=cols[k], ls=lss[j], label=legend_label)

                # Plot the min/max region
                get_ax(i).fill_between(x_vals[:epoch_limit], min_arr[:epoch_limit], max_arr[:epoch_limit], color=cols[k], alpha=0.25)

                # Mark where each experiment completed training (i.e., the early termination condition was met)
                if show_terminals:
                    terminal_points = np.sort(np.argmax(np.isnan(comb_arr), axis=0))[1:]
                    get_ax(i).scatter(x_vals[terminal_points], mean_arr[terminal_points], color='r', s=terminals_size, zorder=1000+i)

                # Plot the inset figure - zooming in on whatever range of the main plot has been specified
                if ins_range:
                    ins.plot(x_vals, mean_arr, c=cols[k], ls=lss[j], label=legend_label)
                    ins.fill_between(x_vals, min_arr, max_arr, color=cols[k], alpha=0.25)
                    ins.set_xlim(ins_x_min, ins_x_max)
                    ins.set_ylim(ins_y_min, ins_y_max)

        get_ax(i).set_xlabel(x_label)
        get_ax(i).set_ylabel(y_label)
        get_ax(i).set_ylim(y_lim)
        get_ax(i).legend(loc=loc, prop={'size': legend_font_size})

    # If there is an inset, mark the region it covers on the main graph
    if ins_loc:
        mark_inset(get_ax(None), ins, loc1=1, loc2=3, fc="none", lw=2, ec='r')

    if save_path is not None:
        fig.savefig(os.path.join(FIG_DIR, save_path), pad_inches=0.2, bbox_inches='tight')


def plot_min_max_logvars(exp_set_label_collection: list):
    _, ax = plt.subplots(1, 1, figsize=(20,10))

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
    _, ax = plt.subplots(1, 1, figsize=(20,10))

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

def plot_evaluation_returns(
    exps: list, title: str = None, xmin=None, xmax=None, ymin=None, ymax=None, feature: str = 'evaluation/return-average', fig_size=(20,10), labels=None, legend=True, save_path=None, erm_dashed=True, loc='upper left', ncol=1,
    font_size=20, legend_font_size=None
    ):
    plt.rc('font', size=font_size)
    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    for i, exp in enumerate(exps):
        if labels is not None:
            label = labels[i]
        else:
            label = f'{exp.name} - {exp.dataset} - REx: {exp.rex} - Beta: {exp.rex_beta} - Seed: {exp.seed}'

        # If `erm_dashed` is True then use dotted lines for experiments using environment models trained without REx
        if erm_dashed:
            ls='--' if not exp.rex else '-'
        else:
            ls='-'

        ax.plot(
            exp.sac.result['timesteps_total'], exp.sac.result[feature], c=cols[i], ls=ls, label=label
        )

        # if feature == 'model/mean_penalty':
        #     ax.axhline(exp.min_penalty, ls=':', c=cols[i])
        #     ax.axhline(exp.max_penalty, ls=':', c=cols[i])

    ax.set_xlabel('Training Steps')
    ax.set_ylabel(FEATURE_LABEL_DICT[feature])
    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(bottom=ymin, top=ymax)
    
    if title is not None:
        ax.set_title(title)
    
    if legend:
        ax.legend(loc=loc, ncol=ncol, prop={'size': legend_font_size or font_size})

    if save_path is not None:
        fig.savefig(os.path.join(FIG_DIR, save_path), pad_inches=0.2, bbox_inches='tight')

def plot_grouped_evaluation_returns(
        exp_set_labels: list, title: str = None, xmin=-1000, ymin=None, ymax=None, show_ends=True, feature: str = 'evaluation/return-average', save_path=None, loc='upper left', max_timestep=500000, end_point_val=-2000,
        font_size=20, legend_font_size=None
    ):
    fig, ax = plt.subplots(1, 1, figsize=(18,10))
    plt.rcParams.update({'font.size': font_size})

    summary_metrics = {}
    for i, (exp_set, label) in enumerate(exp_set_labels):
        exp_results = []
        exp_end_points = []

        for exp in exp_set:
            exp_timesteps = exp.sac.result['timesteps_total'].values

            # Zoom in on the start of the training - most commonly used when looking at the results for an experiment
            # that is currently running.
            exp_timesteps = exp_timesteps[exp_timesteps<=max_timestep]

            # If an experiment fails, the Ray/MOPO code will attempt to re-run it three times.
            # Identify each time an experiment starts.
            # Note: the zeroth element returned by np.where contains an array of indexes; I'm not only selecting the first index.
            # Additionally adding the index of the final step (+1), which is used below when identifying the end points of training attempts.
            start_points = np.hstack([np.where(exp_timesteps==1000)[0], len(exp_timesteps)])
            # The end points are one step before the start points, excluding the very first start point (i.e., 0)
            exp_end_points.append(exp_timesteps[start_points[1:]-1])

            # Only plot the result for the experiment that ran for the longest time.
            # We will mark on the plot where the other experiment attempts (those that failed) terminated.
            # `if` condition appears to be redundant - given we include the start and end of the experiment, len(start_points) will always be > 1.
            if len(start_points) > 1:
                # Identify the length of each attempt
                training_lengths = np.ediff1d(start_points)
                # Identify the start and end index of the longest running experiment attempt
                training_start_ind = np.argmax(training_lengths)
                training_start = start_points[training_start_ind]
                training_end = len(exp_timesteps) if training_start_ind == len(start_points) else start_points[training_start_ind+1]

                # Record the results for the current experiment
                exp_results.append(exp.sac.result[feature][training_start:training_end])
        
        # Given some experiments might fail, the results arrays may be of differing lengths.
        # Fill the end of shorter arrays with np.NaN so that we can stack them.
        comb_arr = np.vstack(list(zip_longest(*[
            exp_res for exp_res in exp_results
            ], fillvalue=np.NaN
        )))
        exp_end_points = np.hstack(exp_end_points)

        # Use the np.nanX methods to ignore the np.NaN values that were added in the previous step
        mean_arr = np.nanmean(comb_arr, axis=-1)
        min_arr = np.nanmin(comb_arr, axis=-1)
        max_arr = np.nanmax(comb_arr, axis=-1)
        std_arr = np.nanstd(comb_arr, axis=-1)
        count_arr = np.sum(~np.isnan(comb_arr), axis=-1)
        x_vals = np.arange(len(mean_arr))*1000

        # Plot the mean values
        ax.plot(x_vals, mean_arr, c=cols[i], label=label)

        # Plot the region showing the standard deviation
        # ax.fill_between(x_vals, min_arr, max_arr, color=cols[i], alpha=0.5)
        ax.fill_between(x_vals, mean_arr-std_arr, mean_arr+std_arr, color=cols[i], alpha=0.5)

        # Identify where experiments had ended and mark these on the plot
        terminal_points = np.sort(np.isnan(comb_arr).argmax(axis=0))
        terminal_inds = terminal_points[np.where(terminal_points>0)[0]]
        ax.scatter(x_vals[terminal_inds], mean_arr[terminal_inds], color=cols[i], edgecolor='k', s=100, zorder=1000+i)

        # If experiments failed, mark the timestamps of failure on the plot
        # These are plotted as crosses below the main plot
        if show_ends:
            exp_end_points = exp_end_points[exp_end_points<x_vals[-1]]
            ax.scatter(exp_end_points, end_point_val*np.ones_like(exp_end_points), marker='x', color=cols[i], s=100)
        
        # Print summary statistics of the results
        # Note that we only consider the evaluation results of the final policy - this is the one we end up with
        summary_metrics[label] = {
            'mean': np.round(mean_arr[-1],0).astype(int),
            'std': np.round(std_arr[-1],0).astype(int),
            'count': count_arr[-2],
            'text': f'{np.round(mean_arr[-1],0).astype(int)} ± {np.round(std_arr[-1],0).astype(int)}'
        }

    ax.set_xlabel('Training Steps')
    ax.set_ylabel(FEATURE_LABEL_DICT[feature])

    if title is not None:
        ax.set_title(title)

    ax.legend(loc=loc, prop={'size': legend_font_size or font_size})

    ax.set_xlim(left=xmin, right=max_timestep+1000)
    ax.set_ylim(bottom=ymin, top=ymax)

    if save_path is not None:
        fig.savefig(os.path.join(FIG_DIR, save_path), pad_inches=0.2, bbox_inches='tight')

    return summary_metrics

def plot_grouped_evaluation_and_dynamics_returns(exp_set_labels: list, title: str = None, xmin=-1000, ymin=None, ymax=None, show_ends=True, save_path=None, loc='upper left', max_timestep=500000, end_point_val=-2000, fig_size=(18,10)):
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    plt.rcParams.update({'font.size': 18})

    summary_metrics = {}
    for i, (exp_set, label) in enumerate(exp_set_labels):
        exp_eval_results = []
        exp_dyn_results = []
        exp_end_points = []

        for exp in exp_set:
            exp_timesteps = exp.sac.result['timesteps_total'].values

            # Zoom in on the start of the training - most commonly used when looking at the results for an experiment
            # that is currently running.
            exp_timesteps = exp_timesteps[exp_timesteps<=max_timestep]

            # If an experiment fails, the Ray/MOPO code will attempt to re-run it three times.
            # Identify each time an experiment starts.
            # Note: the zeroth element returned by np.where contains an array of indexes; I'm not only selecting the first index.
            # Additionally adding the index of the final step (+1), which is used below when identifying the end points of training attempts.
            start_points = np.hstack([np.where(exp_timesteps==1000)[0], len(exp_timesteps)])
            # The end points are one step before the start points, excluding the very first start point (i.e., 0)
            exp_end_points.append(exp_timesteps[start_points[1:]-1])

            # Only plot the result for the experiment that ran for the longest time.
            # We will mark on the plot where the other experiment attempts (those that failed) terminated.
            # `if` condition appears to be redundant - given we include the start and end of the experiment, len(start_points) will always be > 1.
            if len(start_points) > 1:
                # Identify the length of each attempt
                training_lengths = np.ediff1d(start_points)
                # Identify the start and end index of the longest running experiment attempt
                training_start_ind = np.argmax(training_lengths)
                training_start = start_points[training_start_ind]
                training_end = len(exp_timesteps) if training_start_ind == len(start_points) else start_points[training_start_ind+1]

                # Record the results for the current experiment
                exp_eval_results.append(exp.sac.result['evaluation/return-average'][training_start:training_end])
                exp_dyn_results.append(exp.sac.result['model/eval_return_mean'][training_start:training_end])
        
        # Given some experiments might fail, the results arrays may be of differing lengths.
        # Fill the end of shorter arrays with np.NaN so that we can stack them.
        comb_eval_arr = np.vstack(list(zip_longest(*[
            exp_res for exp_res in exp_eval_results
            ], fillvalue=np.NaN
        )))
        comb_dyn_arr = np.vstack(list(zip_longest(*[
            exp_res for exp_res in exp_dyn_results
            ], fillvalue=np.NaN
        )))
        exp_end_points = np.hstack(exp_end_points)

        # Use the np.nanmean methods to ignore the np.NaN values that were added in the previous step
        mean_eval_arr = np.nanmean(comb_eval_arr, axis=-1)
        mean_dyn_arr = np.nanmean(comb_dyn_arr, axis=-1)
        x_vals = np.arange(len(mean_eval_arr))*1000

        # Plot the mean values
        ax.plot(x_vals, mean_eval_arr, c=cols[i], label=f'{label} - Real')
        ax.plot(x_vals, mean_dyn_arr, c=cols[i], ls='--', label=f'{label} - Learned')

        # Plot is too noisy with this turned on
        # ax.fill_between(x_vals, min_arr, max_arr, color=cols[i], alpha=0.5)
        # ax.fill_between(x_vals, mean_arr-std_arr, mean_arr+std_arr, color=cols[i], alpha=0.5)

        # Identify where experiments had ended and mark these on the plot
        terminal_points = np.sort(np.isnan(comb_eval_arr).argmax(axis=0))
        terminal_inds = terminal_points[np.where(terminal_points>0)[0]]
        ax.scatter(x_vals[terminal_inds], mean_eval_arr[terminal_inds], color=cols[i], edgecolor='k', s=100, zorder=1000+i)

        # If experiments failed, mark the timestamps of failure on the plot
        # These are plotted as crosses below the main plot
        if show_ends:
            exp_end_points = exp_end_points[exp_end_points<x_vals[-1]]
            ax.scatter(exp_end_points, end_point_val*np.ones_like(exp_end_points), marker='x', color=cols[i], s=100)
        
        # Print summary statistics of the results
        # Note that we only consider the evaluation results of the final policy - this is the one we end up with
        exploit_diff = mean_dyn_arr-mean_eval_arr
        exploit_diff_mean = np.round(np.nanmean(exploit_diff),0).astype(int)
        exploit_diff_std = np.round(np.nanstd(exploit_diff),0).astype(int)
        summary_metrics[label] = {
            'mean_eval_std': np.round(np.nanstd(mean_eval_arr),0).astype(int),
            'mean_dyn_std': np.round(np.nanstd(mean_dyn_arr),0).astype(int),
            'outliers': np.round(np.sum(np.abs(mean_dyn_arr)>20000)/len(mean_dyn_arr),3),
            'frac_inf': np.round(np.sum(np.isinf(mean_dyn_arr))/len(mean_dyn_arr),3),
            'diff_mean': exploit_diff_mean,
            'diff_std': exploit_diff_std,
            'text': f'{exploit_diff_mean} ± {exploit_diff_std}'
        }
        
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Episode Return')

    if title is not None:
        ax.set_title(title)

    ax.legend(loc=loc, ncol=2)

    ax.set_xlim(left=xmin, right=max_timestep+1000)
    ax.set_ylim(bottom=ymin, top=ymax)

    if save_path is not None:
        fig.savefig(os.path.join(FIG_DIR, save_path), pad_inches=0.2, bbox_inches='tight')

    return summary_metrics

def plot_dynamics_score_vs_agent_return(
    exp_groups, eval_ds, train_eval_ds, agent_score_timesteps=500000, xmin=None, xmax=None, ymin1=None, ymax1=None, primary_metric='log_prob', secondary_metric='overall_mse', show_secondary_metric=True, mode='ood', save_path=None, fig_size=(10,10)
):
    """Note that `dynamics_score` refers to the MSE and/or log-likelihood of the environment model. These could be the ID or OOD scores, or the combined score (i.e., considering both ID and OOD datasets.)
    """
    fig, ax1 = plt.subplots(1, 1, figsize=fig_size)

    if show_secondary_metric:
        ax2 = ax1.twinx()

    scores_all = []
    primary_metric_vals_all = []
    for i, (exps, label) in enumerate(exp_groups):
        scores = []
        primary_metric_vals = []
        secondary_metric_vals = []
        for exp_name in exps:
            exp_results = get_results(exp_name)

            # Look at the results for the specified timestamp
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

    if save_path is not None:
        fig.savefig(os.path.join(FIG_DIR, save_path), pad_inches=0.2, bbox_inches='tight')
