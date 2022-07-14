# import os
# import pickle

# import numpy as np

# from sklearn.decomposition import PCA

# STATE_DIMS = 17
# ACTION_DIMS = 6
# STATE_ACTION_DIMS = STATE_DIMS + ACTION_DIMS
# DATA_DIR = '../dogo_results/data/'
# PCA_DIR = '../dogo_results/pca'

# pca_training_datasets = [
#     'D3RLPY-RT-0.1M-1-P0_1000000.npy',
#     'D3RLPY-RT-0.2M-1-P0_1000000.npy',
#     'D3RLPY-RT-0.5M-1-P0_1000000.npy',
#     'D3RLPY-RT-1M-1-P0_1000000.npy',
#     'SAC-RT-0.1M-0-P0_1000000.npy',
#     'SAC-RT-0.25M-1-P0_1000000.npy',
#     'SAC-RT-0.5M-1-P0_1000000.npy',
#     'SAC-RT-1M-1-P0_1000000.npy',
#     'SAC-RT-2M-1-P0_1000000.npy',
#     'SAC-RT-3M-1-P0_1000000.npy',
#     'D4RL-HC-M.npy',
#     'D4RL-HC-ME.npy',
#     'D4RL-HC-MR.npy',
#     'D4RL-HC-R.npy',
#     'RAND-1.npy',
#     'RAND-2.npy',
#     'RAND-3.npy',
#     'RAND-4.npy',
#     'RAND-5.npy',
#     'RAND-6.npy',
#     'RAND-7.npy',
#     'RAND-8.npy',
#     'RAND-9.npy',
#     'RAND-10.npy',
# ]

# data = np.vstack([np.load(os.path.join(DATA_DIR, tds))[:,:STATE_ACTION_DIMS] for tds in pca_training_datasets])

# ####
# # 1D
# ####
# pca_1 = PCA(1)
# pca_1.fit(data)
# print(pca_1.explained_variance_ratio_)

# with open(os.path.join(PCA_DIR, 'pca_1d.pkl'), 'wb') as f:
#     pickle.dump(pca_1, f)
# with open(os.path.join(PCA_DIR, 'pca_1d_training_datasets.txt'), 'w') as f:
#     f.writelines(pca_training_datasets)
# with open(os.path.join(PCA_DIR, 'pca_1d_explained_var_ratio.txt'), 'w') as f:
#     f.write(str(pca_1.explained_variance_ratio_))

# ####
# # 2D
# ####
# pca_2 = PCA(2)
# pca_2.fit(data)
# print(pca_2.explained_variance_ratio_)

# with open(os.path.join(PCA_DIR, 'pca_2d.pkl'), 'wb') as f:
#     pickle.dump(pca_2, f)
# with open(os.path.join(PCA_DIR, 'pca_2d_training_datasets.txt'), 'w') as f:
#     f.writelines(pca_training_datasets)
# with open(os.path.join(PCA_DIR, 'pca_2d_explained_var_ratio.txt'), 'w') as f:
#     f.write(str(pca_2.explained_variance_ratio_))
