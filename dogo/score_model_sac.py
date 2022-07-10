import sys

from dogo.score_model import score_model

DATA_PATHS = [
    "SAC-RT-0.1M-0-P0_100000.npy",
    "SAC-RT-0.25M-1-P0_100000.npy",
    "SAC-RT-0.5M-1-P0_100000.npy",
    "SAC-RT-1M-1-P0_100000.npy",
    "SAC-RT-2M-1-P0_100000.npy",
    "SAC-RT-3M-1-P0_100000.npy",
]

if __name__ == "__main__":
    experiment = sys.argv[1]
    score_model(experiment, DATA_PATHS)
