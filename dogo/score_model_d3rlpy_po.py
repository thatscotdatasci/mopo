import sys

from dogo.score_model import score_model

DATA_PATHS = [
    "D3RLPY-RT-PO-3,9-0.1M-1-P0_100000.npy",
    "D3RLPY-RT-PO-3,9-0.2M-1-P0_100000.npy",
    "D3RLPY-RT-PO-3,9-0.5M-1-P0_100000.npy",
    "D3RLPY-RT-PO-3,9-1M-1-P0_100000.npy",
    "D3RLPY-RT-PO-3,9-2M-1-P0_100000.npy",
]

if __name__ == "__main__":
    experiment = sys.argv[1]
    score_model(experiment, DATA_PATHS)
