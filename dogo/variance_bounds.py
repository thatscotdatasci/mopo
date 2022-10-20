import os

import numpy as np
from scipy.io import loadmat

from dogo.results import get_experiment_details

EXPERIMENT = 'MP339'
PARAMETERS_PATH = os.path.expanduser("~/rds/rds-dsk-lab-eWkDxBhxBrQ/dimorl/code/mopo/dogo/bnn_params.json")


#############################################################################################
# The MOPO environment model training loss function includes a term for learning
# limits to the variance predictions that are made.
# This is imporant as MOPO penalises reward predictions using the the standard deviation
# of the predictive distribution.
# This simple script returns the learned variance bounds and the max/min L2 norm of the
# standard deviation bounds. The max norm corresponds to the maximum possible reward penalty.
#############################################################################################


def variance_bounds():
    #######################################
    # Load the dynamics model
    # Determine the learned variance bounds
    #######################################
    dynamics_exp_details = get_experiment_details(EXPERIMENT, get_elites=True)
    dynamics_model_dir = os.path.join(dynamics_exp_details.results_dir, 'models')
    params_dict = loadmat(os.path.join(dynamics_model_dir, "BNN_0.mat"))
    min_vars, max_vars = np.exp(params_dict['15']), np.exp(params_dict['14'])
    
    print(f'Min Learned Vars (one per dimension): {min_vars}')
    print(f'Max Learned Vars (one per dimension): {max_vars}')

    print(f'Mean Min Learned Vars: {min_vars.mean()}')
    print(f'Mean Max Learned Vars: {max_vars.mean()}')

    print(f'Max Learned Std L2 Norm: {np.linalg.norm(np.sqrt(max_vars))}')
    print(f'Min Learned Std L2 Norm: {np.linalg.norm(np.sqrt(min_vars))}')


if __name__ == '__main__':
    variance_bounds()
