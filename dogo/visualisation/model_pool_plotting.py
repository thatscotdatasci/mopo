import numpy as np
import matplotlib.pyplot as plt

from dogo.rollouts.split import split_halfcheetah_v2_trans_arr
from dogo.pca.project import learn_project_arr_2d
from dogo.results import PoolArrs, get_sac_pools
from dogo.constants import STATE_DIMS, ACTION_DIMS


def model_pool_learn_pca(exp_list_label_set):
    """ Learn a PCA projection matrix using the model pool samples of the experiments passed.
    """
    exp_lists = (i[0] for i in exp_list_label_set)
    exps = (item for sublist in exp_lists for item in sublist)
    results_arr = (PoolArrs(*get_sac_pools(exp, subsample_size=None)) for exp in exps)
    sa_arr = np.vstack([i.pool[:,:STATE_DIMS+ACTION_DIMS] for i in results_arr])
    return learn_project_arr_2d(sa_arr)

def _model_pool_2dhist(exp_list_label_set, mode, vmin=None, vmax=None, pen_coeff=None, pool=None, pca_model=None, results_arr=None, mean=True, fig_size=None):
    """Create a weighted histogram. The values plotted depend on the `mode` passed, and whether the `mean` flag is True.
    This function should not be called directly - instead, call one of the wrapper functions provided below.
    """

    plt.rcParams.update({'font.size': 22})

    # Get the model pool samples of the experiments passed.
    if results_arr is None:
        exp_lists = [i[0] for i in exp_list_label_set]
        exps = [item for sublist in exp_lists for item in sublist]
        results_arr = {exp: PoolArrs(*get_sac_pools(exp, pool=pool, subsample_size=None, pca_model=pca_model)) for exp in exps}

    n_rows = len(exp_list_label_set)
    n_cols = len(exp_list_label_set[0][0])

    # Bins to be used by the histogram
    bin_vals = np.linspace(-40,40,250)

    # Figure setup
    fig_size = fig_size or (n_cols*10, n_rows*10)
    fig, ax = plt.subplots(n_rows, n_cols, figsize=fig_size)

    # `hist_arrs` will store the histogram array for each experiment
    hist_arrs = []

    # `sum_vals` stores a summary statistic for each experiment
    # this is displayed in the title of the experiment's plot
    sum_vals = []

    # Keep track of the minimum and maximum values encountered
    # Need to ensure that all histograms use the same colourscale
    min_val = np.inf
    max_val = -np.inf

    # First loop through the experiments to generate the results to be plotted
    # Also need to find those minumum and maximum values before we can actually plot the results
    for i, (exp_list, label) in enumerate(exp_list_label_set):
        for j, exp in enumerate(exp_list):
            if results_arr[exp] is None:
                # If there are no results for a given experiment, then record `None` and move on
                hist_arrs.append(None)
                sum_vals.append(None)
                continue

            _, _, _, rew, _, _, pen = split_halfcheetah_v2_trans_arr(results_arr[exp].pool)

            # The histogram weights and summary statistic depend on the mode we are running in
            if mode=='visitation':
                weights = np.ones_like(rew).flatten()
                # Added below - want the summary statistic to be the standard deviation of the visitation values
            elif mode=='pen-rewards':
                weights = rew.flatten()
                sum_vals.append(rew.sum()/len(rew))  # Mean penalised reward
            elif mode=='unpen-rewards':
                weights = (rew+pen_coeff*pen).flatten()
                sum_vals.append((rew+pen_coeff*pen).sum()/len(rew))  # Mean unpenalised reward
            elif mode=='penalties':
                weights = pen.flatten()
                sum_vals.append(pen.sum()/len(rew))  # Mean penalty
            elif mode=='rmse':
                weights = np.sqrt(results_arr[exp].mse_results).flatten()
                sum_vals.append(np.nanmean(np.sqrt(results_arr[exp].mse_results)))  # Mean RMSE
            
            hist_arr, _, _ = np.histogram2d(results_arr[exp].pca_sa_2d[:,0], results_arr[exp].pca_sa_2d[:,1], weights=weights, bins=bin_vals)
            
            # If the `mean` flag is true then divide the weighted histogram values by the number of records in each bucket
            if mode != 'visitation' and mean:
                # This is just the visitation histogram
                counts_arr, _, _ = np.histogram2d(results_arr[exp].pca_sa_2d[:,0], results_arr[exp].pca_sa_2d[:,1], bins=bin_vals)

                # Use the visitation histogram to obtain an array of average weighted values
                hist_arr = np.where(counts_arr==0., 0., hist_arr/counts_arr)
            
            # Add the results to the growing array, and determine if we've encountered a new min/max value
            hist_arrs.append(hist_arr)
            min_val = min(hist_arr.min(), min_val)
            max_val = max(hist_arr.max(), max_val)

            # Summary statistic if we're in 'visitation' mode
            if mode=='visitation':
                sum_vals.append(hist_arr.std())

    # In this second loop we plot the results obtained in the previous loop
    for i, (exp_list, label) in enumerate(exp_list_label_set):
        for j, exp in enumerate(exp_list):
            # Ignore any `None` values encountered
            if results_arr[exp] is None:
                ax[i,j].set_aspect('equal')
                ax[i,j].set_title(f'{exp} - Beta: {label} ')
                continue

            # Plot the histogram
            im = ax[i,j].imshow(
                hist_arrs[i*n_cols+j].T,
                origin='lower',
                vmin=vmin if vmin is not None else min_val,
                vmax=vmax if vmax is not None else max_val,
                extent=[-40,40,-40,40],
                cmap='plasma'
            )

            # Indicate the center of the histgram
            ax[i,j].axhline(0, color='k', ls='--')
            ax[i,j].axvline(0, color='k', ls='--')

            # The title to be used might vary, based on where/how the figure will be used
            # plt_title = f'{exp} - Beta: {label} '
            plt_title = f'{label} - Seed {j+1}\n'

            if mode == 'visitation':
                # plt_title += f'Standard Deviation: {sum_vals[i*n_cols+j]:,.2f}\nExplained Variance: {results_arr[exp].explained_var_2d:.2f}'
                plt_title += f'Explained Variance: {results_arr[exp].explained_var_2d:.2f}'
            elif mode in ['pen-rewards', 'unpen-rewards']:
                plt_title += f'Normalised Reward Sum: {sum_vals[i*n_cols+j]:,.2f}'
            elif mode == 'penalties':
                plt_title += f'Normalised Penalty Sum: {sum_vals[i*n_cols+j]:,.2f}'
            elif mode == 'rmse':
                plt_title += f'Mean RMSE: {sum_vals[i*n_cols+j]:,.2f}'
            
            ax[i,j].set_title(plt_title)
            ax[i,j].set_xlabel('First Principle Component')
            ax[i,j].set_ylabel('Second Principle Component')

    plt.colorbar(im, ax=ax.ravel().tolist())
    return fig


def model_pool_visitation_2dhist(exp_list_label_set, vmin=None, vmax=None, pool=None, pca_model=None, results_arr=None, mean=True):
    return _model_pool_2dhist(exp_list_label_set, mode='visitation', vmin=vmin, vmax=vmax, pool=pool, pca_model=pca_model, results_arr=results_arr, mean=mean)

def model_pool_pen_rewards_2dhist(exp_list_label_set, vmin=None, vmax=None, pool=None, pca_model=None, results_arr=None, mean=True):
    return _model_pool_2dhist(exp_list_label_set, mode='pen-rewards', vmin=vmin, vmax=vmax, pool=pool, pca_model=pca_model, results_arr=results_arr, mean=mean)

def model_pool_unpen_rewards_2dhist(exp_list_label_set, vmin=None, vmax=None, pen_coeff=None, pool=None, pca_model=None, results_arr=None, mean=True):
    return _model_pool_2dhist(exp_list_label_set, mode='unpen-rewards', vmin=vmin, vmax=vmax, pen_coeff=pen_coeff, pool=pool, pca_model=pca_model, results_arr=results_arr, mean=mean)

def model_pool_penalties_2dhist(exp_list_label_set, vmin=None, vmax=None, pool=None, pca_model=None, results_arr=None, mean=True):
    return _model_pool_2dhist(exp_list_label_set, mode='penalties', vmin=vmin, vmax=vmax, pool=pool, pca_model=pca_model, results_arr=results_arr, mean=mean)

def model_pool_rmse_2dhist(exp_list_label_set, vmin=None, vmax=None, pool=None, pca_model=None, results_arr=None, mean=True):
    return _model_pool_2dhist(exp_list_label_set, mode='rmse', vmin=vmin, vmax=vmax, pool=pool, pca_model=pca_model, results_arr=results_arr, mean=mean)
