import pickle

from sklearn.decomposition import PCA

from dogo.constants import PCA_1D, PCA_2D

def project_arr(arr):
    with open(PCA_1D, 'rb') as f:
        pca_1d = pickle.load(f)
    state_action_pca_1d = pca_1d.transform(arr)

    with open(PCA_2D, 'rb') as f:
        pca_2d = pickle.load(f)
    state_action_pca_2d = pca_2d.transform(arr)
    
    return state_action_pca_1d, state_action_pca_2d
