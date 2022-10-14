import os
from itertools import combinations, product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dogo.results import get_experiment_details

SEED = 1443
SCORES_BASE_DIR = "/home/ajc348/rds/rds-dsk-lab-eWkDxBhxBrQ/dimorl/code/dogo_results/mopo/model_scoring"


def experiment_details(exp):
    exp_details = get_experiment_details(exp, get_elites=True)
    model_dir = os.path.join(exp_details.results_dir, 'models')
    elites = exp_details.elites
    scores_dir = os.path.join(SCORES_BASE_DIR, model_dir.split("/")[-3], model_dir.split("/")[-2])
    return scores_dir, elites

def wasserstein_dist(m1, m2, c1, c2):
    return (((m1-m2)**2).sum(axis=-1) + (c1+c2-2*(c1*c2)**0.5).sum(axis=-1)).mean()

def experiment_distances(exp_1, exp_2, datasets):
    scores_dir_1, elites_1 = experiment_details(exp_1)
    scores_dir_2, elites_2 = experiment_details(exp_2)

    wd_vals = []
    for dataset in datasets:
        d1_m = np.load(os.path.join(scores_dir_1, f'{dataset}_{SEED}_means.npy'))[elites_1,:,:]
        d1_c = np.load(os.path.join(scores_dir_1, f'{dataset}_{SEED}_vars.npy'))[elites_1,:,:]

        d2_m = np.load(os.path.join(scores_dir_2, f'{dataset}_{SEED}_means.npy'))[elites_2,:,:]
        d2_c = np.load(os.path.join(scores_dir_2, f'{dataset}_{SEED}_vars.npy'))[elites_2,:,:]
    
        wd_vals.append(wasserstein_dist(d1_m, d2_m, d1_c, d2_c))
    
    return np.array(wd_vals)

def experiment_combination_distances(exps, datasets):
    combs = combinations(exps, 2)
    wd_resulst = []
    for exp_1, exp_2 in combs:
        wd_resulst.append(experiment_distances(exp_1, exp_2, datasets))
    return np.vstack(wd_resulst)

def experiment_product_distances(exps_1, exps_2, datasets):
    prod = product(exps_1, exps_2)
    wd_resulst = []
    for exp_1, exp_2 in prod:
        wd_resulst.append(experiment_distances(exp_1, exp_2, datasets))
    return np.vstack(wd_resulst)

def experiment_collection_distances(experiment_dict):
    datasets, exps = list(experiment_dict.keys()), list(experiment_dict.values())
    n_datasets = len(datasets)

    mean_arr = np.zeros([n_datasets]*3)
    std_arr = np.zeros([n_datasets]*3)
    for i in range(n_datasets):
        for j in range(0, i+1):
            exps_1 = exps[i]
            exps_2 = exps[j]
            if i == j:
                res = experiment_combination_distances(exps_1, datasets)
            else:
                res = experiment_product_distances(exps_1, exps_2, datasets)
            mean_arr[:,i,j] = np.mean(res, axis=0)
            std_arr[:,i,j] = np.std(res, axis=0)
    mean_arr = mean_arr + np.triu(mean_arr.swapaxes(1,2), k=1)
    std_arr = std_arr + np.triu(std_arr.swapaxes(1,2), k=1)
    return mean_arr, std_arr

############################################################################
# Original functions, which hold the model constant and measure the distance
# between predictive distributions for different datasets.
############################################################################

def datset_distances(exp, datasets):
    scores_dir, elites = experiment_details(exp)

    n_datasets = len(datasets)
    wd_vals_arr = np.zeros((n_datasets, n_datasets))
    for i in range(n_datasets):
        dataset_1 = datasets[i]
        d1_m = np.load(os.path.join(scores_dir, f'{dataset_1}_{SEED}_means.npy'))[elites,:,:]
        d1_c = np.load(os.path.join(scores_dir, f'{dataset_1}_{SEED}_vars.npy'))[elites,:,:]
        for j in range(i+1):
            dataset_2 = datasets[j]
            d2_m = np.load(os.path.join(scores_dir, f'{dataset_2}_{SEED}_means.npy'))[elites,:,:]
            d2_c = np.load(os.path.join(scores_dir, f'{dataset_2}_{SEED}_vars.npy'))[elites,:,:]
            wd_vals_arr[i,j] = wasserstein_dist(d1_m, d2_m, d1_c, d2_c)
        print(f'Completed {i+1}/{n_datasets} datasets')
    return wd_vals_arr + wd_vals_arr.T - np.diag(wd_vals_arr.diagonal())

def distances_plot(wd_vals_arr, datasets):
    fig, ax = plt.subplots(1, 1, figsize=(len(datasets),len(datasets)))

    mat = ax.matshow(wd_vals_arr)
    ax.set_xticks(range(len(datasets)))
    ax.set_yticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=45)
    ax.set_yticklabels(datasets, rotation=45)

    for (i,j), z in np.ndenumerate(wd_vals_arr):
        if z != 0:
            ax.text(j, i, '{0:.2f}'.format(z), ha="center", va="center", color='w' if i == j else 'k')

    fig.colorbar(mat)

    return fig

def wasserstein_distances(exp, datasets, save_output=True, return_plot=False):
    scores_dir, _ = experiment_details(exp)
    
    wd_vals_arr = datset_distances(exp, datasets)
    wd_vals_df = pd.DataFrame(wd_vals_arr, index=datasets, columns=datasets)
    if save_output:
        wd_vals_df.to_json(os.path.join(scores_dir, 'wasserstein_distances.json'))

    wd_plot = distances_plot(wd_vals_arr, datasets)
    if save_output: 
        wd_plot.savefig(os.path.join(scores_dir, 'wasserstein_distances.jpeg'))
    
    if return_plot:
        return wd_vals_arr, wd_plot
    else:
        return wd_vals_arr, None
