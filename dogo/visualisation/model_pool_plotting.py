import numpy as np
import matplotlib.pyplot as plt

from dogo.rollouts.split import split_halfcheetah_v2_trans_arr


def _model_pool_2dhist(results_arr, exp_list_label_set, mode, vmin=None, vmax=None, pen_coeff=None):
    n_rows = len(exp_list_label_set)
    n_cols = len(exp_list_label_set[0][0])
    bin_vals = np.linspace(-40,40,250)

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols*10, n_rows*10))

    hist_arrs = []
    sum_vals = []
    min_val = np.inf
    max_val = -np.inf

    for i, (exp_list, label) in enumerate(exp_list_label_set):
        for j, exp in enumerate(exp_list):
            _, _, _, rew, _, _, pen = split_halfcheetah_v2_trans_arr(results_arr[exp].pool)

            if mode=='visitation':
                weights = np.ones_like(rew).flatten()
                # Added below
            elif mode=='pen-rewards':
                weights = rew.flatten()
                sum_vals.append(rew.sum()/len(rew))
            elif mode=='unpen-rewards':
                weights = (rew+pen_coeff*pen).flatten()
                sum_vals.append((rew+pen_coeff*pen).sum()/len(rew))
            elif mode=='penalties':
                weights = pen.flatten()
                sum_vals.append(pen.sum()/len(rew))
            elif mode=='rmse':
                weights = np.sqrt(results_arr[exp].mse_results).flatten()
                sum_vals.append(np.sqrt(results_arr[exp].mse_results).mean())
            
            hist_arr, _, _ = np.histogram2d(results_arr[exp].pca_sa_2d[:,0], results_arr[exp].pca_sa_2d[:,1], weights=weights, bins=bin_vals)
            if mode != 'visitation':
                counts_arr, _, _ = np.histogram2d(results_arr[exp].pca_sa_2d[:,0], results_arr[exp].pca_sa_2d[:,1], bins=bin_vals)
                hist_arr = np.where(counts_arr==0., 0., hist_arr/counts_arr)
            
            hist_arrs.append(hist_arr)
            min_val = min(hist_arr.min(), min_val)
            max_val = max(hist_arr.max(), max_val)

            if mode=='visitation':
                sum_vals.append(hist_arr.std())

    for i, (exp_list, label) in enumerate(exp_list_label_set):
        for j, exp in enumerate(exp_list):
            im = ax[i,j].imshow(
                hist_arrs[i*n_cols+j].T,
                origin='lower',
                vmin=vmin if vmin is not None else min_val,
                vmax=vmax if vmax is not None else max_val,
                extent=[-40,40,-40,40],
                cmap='plasma'
            )
            ax[i,j].axhline(0, color='k', ls='--')
            ax[i,j].axvline(0, color='k', ls='--')
            plt_title = f'{exp} - Beta: {label} '
            if mode == 'visitation':
                plt_title += f'Standard Deviation: {sum_vals[i*n_cols+j]:,.2f}'
            elif mode in ['pen-rewards', 'unpen-rewards']:
                plt_title += f'Normalised Reward Sum: {sum_vals[i*n_cols+j]:,.2f}'
            elif mode == 'penalties':
                plt_title += f'Normalised Penalty Sum: {sum_vals[i*n_cols+j]:,.2f}'
            elif mode == 'rmse':
                plt_title += f'Mean RMSE: {sum_vals[i*n_cols+j]:,.2f}'
            plt_title += f'\nExplained Variance: {results_arr[exp].explained_var_2d:.2f}'
            ax[i,j].set_title(plt_title)
            ax[i,j].set_xlabel('First Principle Component')
            ax[i,j].set_ylabel('Second Principle Component')

    plt.colorbar(im, ax=ax.ravel().tolist())


def model_pool_visitation_2dhist(results_arr, exp_list_label_set, vmin=None, vmax=None):
    return _model_pool_2dhist(results_arr, exp_list_label_set, mode='visitation', vmin=vmin, vmax=vmax)

def model_pool_pen_rewards_2dhist(results_arr, exp_list_label_set, vmin=None, vmax=None):
    return _model_pool_2dhist(results_arr, exp_list_label_set, mode='pen-rewards', vmin=vmin, vmax=vmax)

def model_pool_unpen_rewards_2dhist(results_arr, exp_list_label_set, vmin=None, vmax=None, pen_coeff=None):
    return _model_pool_2dhist(results_arr, exp_list_label_set, mode='unpen-rewards', vmin=vmin, vmax=vmax, pen_coeff=pen_coeff)

def model_pool_penalties_2dhist(results_arr, exp_list_label_set, vmin=None, vmax=None):
    return _model_pool_2dhist(results_arr, exp_list_label_set, mode='penalties', vmin=vmin, vmax=vmax)

def model_pool_rmse_2dhist(results_arr, exp_list_label_set, vmin=None, vmax=None):
    return _model_pool_2dhist(results_arr, exp_list_label_set, mode='rmse', vmin=vmin, vmax=vmax)
