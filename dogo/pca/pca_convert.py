import os
import sys
import pickle

import numpy as np

from sklearn.decomposition import PCA

STATE_DIMS = 17
ACTION_DIMS = 6
STATE_ACTION_DIMS = STATE_DIMS + ACTION_DIMS
DATA_DIR = '../dogo_results/data/'
PCA_1D = 'pca/pca_1d.pkl'
PCA_2D = 'pca/pca_2d.pkl'

def main(dataset):
    state_action_arr = np.load(os.path.join(DATA_DIR, f'{dataset}.npy'))[:,:STATE_ACTION_DIMS]

    ####
    # 1D
    ####
    with open(PCA_1D, 'rb') as f:
        pca_1d = pickle.load(f)
    state_action_pca_1d = pca_1d.transform(state_action_arr)
    np.save(os.path.join(DATA_DIR, 'pca', f'{dataset}_1d.npy'), state_action_pca_1d)

    ####
    # 2D
    ####
    with open(PCA_2D, 'rb') as f:
        pca_2d = pickle.load(f)
    state_action_pca_2d = pca_2d.transform(state_action_arr)
    np.save(os.path.join(DATA_DIR, 'pca', f'{dataset}_2d.npy'), state_action_pca_2d)

if __name__ == '__main__':
    dataset = sys.argv[1]
    main(dataset)
