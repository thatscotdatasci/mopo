import sys

from dogo.score_model import score_model

DATA_PATHS = [
    "MIXED-3-PO-1.npy",
    "D3RLPY-PO-MP1-P0_20000.npy",
    "D3RLPY-PO-MP1-P1_20000.npy",
    "D3RLPY-PO-MP1-P2_20000.npy",
    "D3RLPY-PO-MP1-P3_20000.npy",
    "D3RLPY-PO-MP1-P4_20000.npy",
    "D3RLPY-PO-MP1-P0_100000.npy",
    "D3RLPY-PO-MP1-P1_100000.npy",
    "D3RLPY-PO-MP1-P2_100000.npy",
    "D3RLPY-PO-MP1-P3_100000.npy",
    "D3RLPY-PO-MP1-P4_100000.npy",
    "D3RLPY-PO-PAP1-P0_20000.npy",
    "D3RLPY-PO-PAP1-P1_20000.npy",
    "D3RLPY-PO-PAP1-P2_20000.npy",
    "D3RLPY-PO-PAP1-P3_20000.npy",
    "D3RLPY-PO-PAP1-P4_20000.npy",
    "D3RLPY-PO-PAP1-P0_100000.npy",
    "D3RLPY-PO-PAP1-P1_100000.npy",
    "D3RLPY-PO-PAP1-P2_100000.npy",
    "D3RLPY-PO-PAP1-P3_100000.npy",
    "D3RLPY-PO-PAP1-P4_100000.npy",
    "RAND-1-PO-1.npy",
    "RAND-2-PO-1.npy",
    "RAND-3-PO-1.npy",
    "RAND-4-PO-1.npy",
    "RAND-5-PO-1.npy",
    "RAND-6-PO-1.npy",
    "RAND-7-PO-1.npy",
    "RAND-8-PO-1.npy",
    "RAND-9-PO-1.npy",
    "RAND-10-PO-1.npy",
    "RAND-D3RLPY-PO-MP1-P0-1_100000.npy",
    "RAND-D3RLPY-PO-MP1-P1-1_100000.npy",
    "RAND-D3RLPY-PO-MP1-P2-1_100000.npy",
    "RAND-D3RLPY-PO-MP1-P3-1_100000.npy",
    "RAND-D3RLPY-PO-MP1-P4-1_100000.npy",
    "RAND-D3RLPY-PO-PAP1-P0-1_100000.npy",
    "RAND-D3RLPY-PO-PAP1-P1-1_100000.npy",
    "RAND-D3RLPY-PO-PAP1-P2-1_100000.npy",
    "RAND-D3RLPY-PO-PAP1-P3-1_100000.npy",
    "RAND-D3RLPY-PO-PAP1-P4-1_100000.npy",
]

if __name__ == "__main__":
    experiment = sys.argv[1]
    score_model(experiment, DATA_PATHS)
