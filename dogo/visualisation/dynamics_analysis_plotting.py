import os

import numpy as np
import matplotlib.pyplot as plt

from dogo.results import get_experiment_details

def retrieve_metric(metric, exp, dataset):
    return np.load(os.path.join('dynamics', f'{exp}_{dataset}_{metric}.npy'), allow_pickle=True)

def plot_metric_2d(experiment_names, metric, dataset):
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    x_vals = np.load(f'/home/ajc348/rds/hpc-work/dogo_results/data/pca/{dataset}_1d.npy')[:10000,:]

    for exp_name in experiment_names:
        exp = get_experiment_details(exp_name)
        vals_arr = retrieve_metric(metric, exp.name, dataset)
        y_vals = vals_arr.mean(axis=1)
        ax.scatter(x_vals, y_vals, marker='x', label=f'{exp.name} - REx: {exp.rex} - Beta: {exp.rex_beta} - Seed: {exp.seed}')

    ax.legend()

def plot_metric_3d(experiment_names, metric, dataset):
    fig, ax = plt.subplots(1, 2, figsize=(20,10), subplot_kw={"projection": "3d"})
    x_vals = np.load(f'/home/ajc348/rds/hpc-work/dogo_results/data/pca/{dataset}_2d.npy')[:10000,:]

    for exp_name in experiment_names:
        exp = get_experiment_details(exp_name)
        vals_arr = retrieve_metric(metric, exp.name, dataset)
        y_vals = vals_arr.mean(axis=1)

        for i in range(2):
            ax[i].scatter(x_vals[:,0], x_vals[:,1], y_vals, marker='x', label=f'{exp.name} - REx: {exp.rex} - Beta: {exp.rex_beta} - Seed: {exp.seed}')

        ax[0].view_init(10, 60)
        ax[1].view_init(90, 0)
    
    for i in range(2):
        ax[i].legend()