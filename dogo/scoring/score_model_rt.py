import sys

from dogo.scoring.score_model import score_model

DATA_PATHS = [
    "D3RLPY-RT-0.1M-4-P0_10000.npy",
    "D3RLPY-RT-0.2M-4-P0_10000.npy",
    "D3RLPY-RT-0.5M-4-P0_10000.npy",
    "D3RLPY-RT-1M-4-P0_10000.npy",
    "D3RLPY-RT-2M-4-P0_10000.npy",
    "D3RLPY-RT-0.1M-2-P0_10000.npy",
    "D3RLPY-RT-0.2M-2-P0_10000.npy",
    "D3RLPY-RT-0.5M-2-P0_10000.npy",
    "D3RLPY-RT-1M-2-P0_10000.npy",
    "D3RLPY-RT-2M-2-P0_10000.npy",
    "D3RLPY-RT-0.1M-3-P0_10000.npy",
    "D3RLPY-RT-0.2M-3-P0_10000.npy",
    "D3RLPY-RT-0.5M-3-P0_10000.npy",
    "D3RLPY-RT-1M-3-P0_10000.npy",
    "D3RLPY-RT-2M-3-P0_10000.npy",
    "D4RL-HC-M_10000.npy",
    "D4RL-HC-ME_10000.npy",
    "D4RL-HC-MR_10000.npy",
    "D4RL-HC-R_10000.npy",
    "SAC-RT-0.1M-4-P0_10000.npy",
    "SAC-RT-0.25M-4-P0_10000.npy",
    "SAC-RT-0.5M-4-P0_10000.npy",
    "SAC-RT-1M-4-P0_10000.npy",
    "SAC-RT-2M-4-P0_10000.npy",
    "SAC-RT-3M-4-P0_10000.npy",
    "SAC-RT-0.25M-2-P0_10000.npy",
    "SAC-RT-0.5M-2-P0_10000.npy",
    "SAC-RT-1M-2-P0_10000.npy",
    "SAC-RT-2M-2-P0_10000.npy",
    "SAC-RT-3M-2-P0_10000.npy",
    "SAC-RT-0.25M-3-P0_10000.npy",
    "SAC-RT-0.5M-3-P0_10000.npy",
    "SAC-RT-1M-3-P0_10000.npy",
    "SAC-RT-2M-3-P0_10000.npy",
    "SAC-RT-3M-3-P0_10000.npy",
    "RAND-1_10000.npy",
    "RAND-2_10000.npy",
    "RAND-3_10000.npy",
]

if __name__ == "__main__":
    experiment = sys.argv[1]
    score_model(experiment, DATA_PATHS)
