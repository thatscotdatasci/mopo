import os
import numpy as np

####################################################################################
# Purpose: 
# Simple script that will combine all the arrays specified in ARRAYS_TO_MERGE into
# a single array. An additional dimension will be added to the end, which indicates
# the policy the data belongs to (i.e. the index of the dataset in ARRAYS_TO_MERGE). 
####################################################################################

ARRAYS_TO_MERGE = [
    "/home/ajc348/rds/hpc-work/softlearning/gym/HalfCheetah/v2/2022-05-17T18-38-36-half_cheetah_v2_3M/id=31acc_00000-seed=9479/short_rollouts/checkpoint_30/rollout_25000_0.npy",
    "/home/ajc348/rds/hpc-work/softlearning/gym/HalfCheetah/v2/2022-05-17T18-38-36-half_cheetah_v2_3M/id=31acc_00000-seed=9479/short_rollouts/checkpoint_60/rollout_25000_0.npy",
    "/home/ajc348/rds/hpc-work/softlearning/gym/HalfCheetah/v2/2022-05-17T18-38-36-half_cheetah_v2_3M/id=31acc_00000-seed=9479/short_rollouts/checkpoint_90/rollout_25000_0.npy",
    "/home/ajc348/rds/hpc-work/softlearning/gym/HalfCheetah/v2/2022-05-17T18-38-36-half_cheetah_v2_3M/id=31acc_00000-seed=9479/short_rollouts/checkpoint_120/rollout_25000_0.npy",
]
OUTPUT_DIR = "/home/ajc348/rds/hpc-work/mopo/rollouts/softlearning/HalfCheetah/v2/2022-05-17T18-38-36-half_cheetah_v2_3M/id=31acc_00000-seed=9479/diverse_policies"

def main():
    # First ensure that all arrays do exist
    for arr_path in ARRAYS_TO_MERGE:
        assert os.path.exists(arr_path)

    # Also ensure the output directory does not already exist
    if os.path.exists(OUTPUT_DIR):
        raise FileExistsError('Please delete output directory before re-running')
    os.makedirs(OUTPUT_DIR)

    # Loop through the arrays to be combined
    # Add a column which indicates the policy the data came from
    for i, arr_path in enumerate(ARRAYS_TO_MERGE):
        # Load data
        trans_arr = np.load(arr_path)

        # The first array dictates the number of columns that should be present
        if i == 0:
            cols = trans_arr.shape[1]
        assert trans_arr.shape[1] == cols

        # Add a column to the array with the ID of the trajectory
        policy_id = np.full((trans_arr.shape[0], 1), i)
        trans_arr = np.hstack((trans_arr, policy_id))

        # Combine the trajectories
        if i == 0:
            final_arr = np.copy(trans_arr)
        else:
            final_arr = np.vstack((final_arr, np.copy(trans_arr)))

    np.save(os.path.join(OUTPUT_DIR, f'combined_transitions.npy'), final_arr)

if __name__ == "__main__":
    main()
