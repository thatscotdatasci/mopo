import sys

from dogo.score_model import score_model

DATA_PATHS = [
    "H-D3RLPY-RT-0.1M-1-P0_10000.npy",
    "H-D3RLPY-RT-0.2M-1-P0_10000.npy",
    "H-D3RLPY-RT-0.4M-1-P0_10000.npy",
    "H-D3RLPY-RT-0.6M-1-P0_10000.npy",
    "H-D3RLPY-RT-0.8M-1-P0_10000.npy",
    "H-D3RLPY-RT-1M-1-P0_10000.npy",
    "H-D3RLPY-RT-0.1M-2-P0_10000.npy",
    "H-D3RLPY-RT-0.2M-2-P0_10000.npy",
    "H-D3RLPY-RT-0.4M-2-P0_10000.npy",
    "H-D3RLPY-RT-0.6M-2-P0_10000.npy",
    "H-D3RLPY-RT-0.8M-2-P0_10000.npy",
    "H-D3RLPY-RT-1M-2-P0_10000.npy",
    "H-D3RLPY-RT-0.1M-3-P0_10000.npy",
    "H-D3RLPY-RT-0.2M-3-P0_10000.npy",
    "H-D3RLPY-RT-0.4M-3-P0_10000.npy",
    "H-D3RLPY-RT-0.6M-3-P0_10000.npy",
    "H-D3RLPY-RT-0.8M-3-P0_10000.npy",
    "H-D3RLPY-RT-1M-3-P0_10000.npy",
    "D4RL-H-M_10000.npy",
    "D4RL-H-ME_10000.npy",
    "D4RL-H-MR_10000.npy",
    "D4RL-H-R_10000.npy",
    "H-SAC-RT-0.1M-1-P0_10000.npy",
    "H-SAC-RT-0.2M-1-P0_10000.npy",
    "H-SAC-RT-0.4M-1-P0_10000.npy",
    "H-SAC-RT-0.6M-1-P0_10000.npy",
    "H-SAC-RT-0.8M-1-P0_10000.npy",
    "H-SAC-RT-1M-1-P0_10000.npy",
    "H-SAC-RT-0.1M-2-P0_10000.npy",
    "H-SAC-RT-0.2M-2-P0_10000.npy",
    "H-SAC-RT-0.4M-2-P0_10000.npy",
    "H-SAC-RT-0.6M-2-P0_10000.npy",
    "H-SAC-RT-0.8M-2-P0_10000.npy",
    "H-SAC-RT-1M-2-P0_10000.npy",
    "H-SAC-RT-0.1M-3-P0_10000.npy",
    "H-SAC-RT-0.2M-3-P0_10000.npy",
    "H-SAC-RT-0.4M-3-P0_10000.npy",
    "H-SAC-RT-0.6M-3-P0_10000.npy",
    "H-SAC-RT-0.8M-3-P0_10000.npy",
    "H-SAC-RT-1M-3-P0_10000.npy",
    "H-RAND-1_10000.npy",
    "H-RAND-2_10000.npy",
    "H-RAND-3_10000.npy",
]

if __name__ == "__main__":
    experiment = sys.argv[1]
    score_model(experiment, DATA_PATHS)
