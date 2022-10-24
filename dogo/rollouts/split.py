from collections import namedtuple

import numpy as np

from dogo.constants import HC_STATE_DIMS, HC_ACTION_DIMS, HC_REWARD_DIMS, HC_TERMINAL_DIMS, HC_POLICY_DIMS

#######################################################
# Dimensions for HalfCheetah (and Walker2d) environment
# DO NOT TRY TO APPLY TO OTHERS WITHOUT CHECKING!
#######################################################
HC_DIMS = [HC_STATE_DIMS, HC_ACTION_DIMS, HC_STATE_DIMS, HC_REWARD_DIMS, HC_TERMINAL_DIMS, HC_POLICY_DIMS]
HC_ENDS = np.cumsum(HC_DIMS)

#######################################################
# Dimensions for Hopper environment
#######################################################
HOPPER_STATE_DIMS = 11
HOPPER_ACTION_DIMS = 3
HOPPER_REWARD_DIMS = 1
HOPPER_TERMINAL_DIMS = 1
HOPPER_POLICY_DIMS = 1
HOPPER_PENALTY_DIMS = 1

HOPPER_DIMS = [HOPPER_STATE_DIMS, HOPPER_ACTION_DIMS, HOPPER_STATE_DIMS, HOPPER_REWARD_DIMS, HOPPER_TERMINAL_DIMS, HOPPER_POLICY_DIMS]
HOPPER_ENDS = np.cumsum(HOPPER_DIMS)


TransRecords = namedtuple('TransRecords', 'states actions next_states rewards dones policies penalties')

##############
# Helper Funcs
##############

# Use the below to split transition records that include policies and MOPO penalties
def split_halfcheetah_v2_trans_arr(arr: str):
    return TransRecords(*np.split(arr, HC_ENDS, axis=1))

def split_hopper_v2_trans_arr(arr: str):
    return TransRecords(*np.split(arr, HOPPER_ENDS, axis=1))

# Use the below to split "normal" transition records, that do not include policies and MOPO penalties
REG_DIMS = [HC_STATE_DIMS, HC_ACTION_DIMS, HC_STATE_DIMS, HC_REWARD_DIMS]
REG_ENDS = np.cumsum(REG_DIMS)
RegRecords = namedtuple('RegRecords', 'states actions next_states rewards dones')
def split_halfcheetah_v2_reg_arr(arr: str):
    return RegRecords(*np.split(arr, REG_ENDS, axis=1))
