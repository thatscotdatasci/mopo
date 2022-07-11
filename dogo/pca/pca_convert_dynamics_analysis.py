import os
import pickle
from glob import glob

import numpy as np

from sklearn.decomposition import PCA

POLICY_DIR = '/home/ajc348/rds/hpc-work/dogo_results/mopo/analysis/policy'
PCA_1D = 'pca/pca_1d.pkl'
PCA_2D = 'pca/pca_2d.pkl'

def main():
    with open(PCA_1D, 'rb') as f:
        pca_1d = pickle.load(f)

    with open(PCA_2D, 'rb') as f:
        pca_2d = pickle.load(f)

    for f in glob(os.path.join(POLICY_DIR, '*_state_action.npy')):
        state_action_arr = np.load(f)
        filename = os.path.split(f)[-1]

        ####
        # 1D
        ####
        state_action_pca_1d = np.stack([pca_1d.transform(i) for i in state_action_arr], axis=-1)
        np.save(os.path.join(POLICY_DIR, filename.replace('.npy', '_pca_1d.npy')), state_action_pca_1d)

        ####
        # 2D
        ####
        state_action_pca_2d = np.stack([pca_2d.transform(i) for i in state_action_arr], axis=-1)
        np.save(os.path.join(POLICY_DIR, filename.replace('.npy', '_pca_2d.npy')), state_action_pca_2d)

if __name__ == '__main__':
    main()
