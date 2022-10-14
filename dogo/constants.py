import os

DEFAULT_SEED = 1443

#######
# Paths
#######
MOPO_RESULTS_MAP_PATH = os.path.expanduser("~/rds/rds-dsk-lab-eWkDxBhxBrQ/dimorl/code/dogo_results/mopo/results_map.json")
MOPO_BASEDIR = os.path.expanduser("~/rds/hpc-work/mopo")
RESULTS_BASEDIR = os.path.expanduser("~/rds/rds-dsk-lab-eWkDxBhxBrQ/dimorl/code/dogo_results/mopo")
SCORING_BASEDIR = os.path.expanduser("~/rds/rds-dsk-lab-eWkDxBhxBrQ/dimorl/code/dogo_results/mopo/model_scoring")
DATA_DIR = os.path.expanduser("~/rds/rds-dsk-lab-eWkDxBhxBrQ/dimorl/code/dogo_results/data")
FIG_DIR = os.path.expanduser("~/rds/hpc-work/mopo/figs")

#######
# Docs
#######
DYNAMICS_TRAINING_FILES = {
    "model_loss_history"                   : "model_loss_history.txt",
    "model_pol_total_loss_history"         : "model_pol_total_loss_history.txt",
    "model_pol_var_loss_history"           : "model_pol_var_loss_history.txt",
    "model_mean_pol_loss_history"          : "model_mean_pol_loss_history.txt",
    "model_holdout_loss_history"           : "model_holdout_loss_history.txt",
    "model_holdout_pol_total_loss_history" : "model_holdout_pol_total_loss_history.txt",
    "model_holdout_pol_var_loss_history"   : "model_holdout_pol_var_loss_history.txt",
    "model_holdout_mean_pol_loss_history"  : "model_holdout_mean_pol_loss_history.txt",
    "model_train_loss_history"             : "model_train_loss_history.txt",
    "model_train_core_loss_history"        : "model_train_core_loss_history.txt",
    "model_train_pol_total_loss_history"   : "model_train_pol_total_loss_history.txt",
    "model_train_pol_var_loss_history"     : "model_train_pol_var_loss_history.txt",
    "model_train_decay_loss_history"       : "model_train_decay_loss_history.txt",
    "model_train_var_lim_loss_history"     : "model_train_var_lim_loss_history.txt",
}
SAC_TRAINING_FILES = {
    "result": "result.json"
}

#####
# PCA
#####
PCA_1D = os.path.expanduser('~/rds/rds-dsk-lab-eWkDxBhxBrQ/dimorl/code/dogo_results/pca/pca_1d.pkl')
PCA_2D = os.path.expanduser('~/rds/rds-dsk-lab-eWkDxBhxBrQ/dimorl/code/dogo_results/pca/pca_2d.pkl')

############
# Dimensions
############
STATE_DIMS = 17
ACTION_DIMS = 6

#######
# Names
#######
D4RL_NAME_DICT = {
    'halfcheetah-medium-expert-v0': 'Medium-Expert',
    'halfcheetah-medium-replay-v0': 'Medium-Replay',
    'halfcheetah-medium-v0': 'Medium',
    'halfcheetah-random-v0': 'Random'
}
