DEFAULT_SEED = 1443

#######
# Paths
#######
MOPO_RESULTS_MAP_PATH = "../dogo_results/mopo/results_map.json"
RESULTS_BASEDIR = "../dogo_results/mopo"
SCORING_BASEDIR = "../dogo_results/mopo/model_scoring"

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
    "model_train_decay_loss_history"       : "model_train_decay_loss_history.txt",
    "model_train_var_lim_loss_history"     : "model_train_var_lim_loss_history.txt",
}
SAC_TRAINING_FILES = {
    "result": "result.json"
}