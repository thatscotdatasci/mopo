import sys

from dogo.score_model import score_model

DATA_PATHS = [
    "D4RL-HC-M_100000.npy",
    "D4RL-HC-ME_100000.npy",
    "D4RL-HC-MR_100000.npy",
    "D4RL-HC-R_100000.npy",
]

if __name__ == "__main__":
    experiment = sys.argv[1]
    score_model(experiment, DATA_PATHS)
