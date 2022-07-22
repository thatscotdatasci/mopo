from collections import namedtuple

import numpy as np

#################################################
# Dimensions for HalfCheetah v2 dataset
# DO NOT TRY TO APPLY TO OTHERS WITHOUT CHECKING!
#################################################
STATE_DIMS = 17
ACTION_DIMS = 6
REWARD_DIMS = 1
TERMINAL_DIMS = 1
POLICY_DIMS = 1
PENALTY_DIMS = 1

DIMS = [STATE_DIMS, ACTION_DIMS, STATE_DIMS, REWARD_DIMS, TERMINAL_DIMS, POLICY_DIMS]
ENDS = np.cumsum(DIMS)

##############
# Helper Funcs
##############
TransRecords = namedtuple('TransRecords', 'states actions next_states rewards dones policies penalties')
def split_halfcheetah_v2_trans_arr(arr: str):
    return TransRecords(*np.split(arr, ENDS, axis=1))

REG_DIMS = [STATE_DIMS, ACTION_DIMS, STATE_DIMS, REWARD_DIMS]
REG_ENDS = np.cumsum(REG_DIMS)
RegRecords = namedtuple('RegRecords', 'states actions next_states rewards dones')
def split_halfcheetah_v2_reg_arr(arr: str):
    return RegRecords(*np.split(arr, REG_ENDS, axis=1))
