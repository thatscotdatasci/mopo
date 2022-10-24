import os
import pickle

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

from dogo.constants import PCA_1D, PCA_2D, HC_STATE_DIMS, HC_ACTION_DIMS


HC_STATE_ACTION_DIMS = HC_STATE_DIMS + HC_ACTION_DIMS
DATA_DIR = os.path.expanduser('~/rds/rds-dsk-lab-eWkDxBhxBrQ/dimorl/code/dogo_results/data')

PCA_BASE_DATASETS = [
    'D3RLPY-RT-0.1M-1-P0_1000000.npy',
    'D3RLPY-RT-0.2M-1-P0_1000000.npy',
    'D3RLPY-RT-0.5M-1-P0_1000000.npy',
    'D3RLPY-RT-1M-1-P0_1000000.npy',
    'SAC-RT-0.1M-0-P0_1000000.npy',
    'SAC-RT-0.25M-1-P0_1000000.npy',
    'SAC-RT-0.5M-1-P0_1000000.npy',
    'SAC-RT-1M-1-P0_1000000.npy',
    'SAC-RT-2M-1-P0_1000000.npy',
    'SAC-RT-3M-1-P0_1000000.npy',
    'D4RL-HC-M.npy',
    'D4RL-HC-ME.npy',
    'D4RL-HC-MR.npy',
    'D4RL-HC-R.npy',
    'RAND-1.npy',
    'RAND-2.npy',
    'RAND-3.npy',
    'RAND-4.npy',
    'RAND-5.npy',
    'RAND-6.npy',
    'RAND-7.npy',
    'RAND-8.npy',
    'RAND-9.npy',
    'RAND-10.npy',
]

def project_arr(arr, pca_1d=None, pca_2d=None):
    """ Project the passed array using either the passed PCA models, or defaults.
    """

    # Transform joint angles to lie in [-pi, pi] range
    # They can lie outside of this range, but this adds a spurious source of variance
    arr[:,1:8] = np.arctan2(np.sin(arr[:,1:8]), np.cos(arr[:,1:8]))

    # If no PCA model was passed, load the default model
    # This was trained on the files stated in:
    # ~/rds/rds-dsk-lab-eWkDxBhxBrQ/dimorl/code/dogo_results/pca/pca_1d_training_datasets.txt
    if pca_1d is None:
        with open(PCA_1D, 'rb') as f:
            pca_1d = pickle.load(f)
    state_action_pca_1d = pca_1d.transform(arr)

    # If no PCA model was passed, load the default model
    # This was trained on the files stated in:
    # ~/rds/rds-dsk-lab-eWkDxBhxBrQ/dimorl/code/dogo_results/pca/pca_2d_training_datasets.txt
    if pca_2d is None:
        with open(PCA_2D, 'rb') as f:
            pca_2d = pickle.load(f)
    state_action_pca_2d = pca_2d.transform(arr)

    # Determine the fraction of the original variance that is captured by the projection
    explained_var_2d = r2_score(arr, pca_2d.inverse_transform(state_action_pca_2d), multioutput='variance_weighted')
    
    return state_action_pca_1d, state_action_pca_2d, explained_var_2d

def learn_project_arr_2d(arr, inc_training_data=False):
    """ Use the passed arr to learn a projection matrix, and then use it to project the arrau.
    If `inc_training_data` is True then the `PCA_BASE_DATASETS` will also be used to train the projection matrix.
    """

    # Transform joint angles to lie in [-pi, pi range]
    # They can lie outside of this range, but this adds a spurious source of variance
    arr[:,1:8] = np.arctan2(np.sin(arr[:,1:8]), np.cos(arr[:,1:8]))

    if inc_training_data:
        training_data = np.vstack([np.load(os.path.join(DATA_DIR, tds))[:,:HC_STATE_ACTION_DIMS] for tds in PCA_BASE_DATASETS])
        arr = np.vstack([training_data, arr])

    pca_2d = PCA(2)
    pca_2d.fit(arr)
    
    return pca_2d
