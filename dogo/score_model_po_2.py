import sys

from dogo.score_model import score_model

DATA_PATHS = [
    "MIXED-3-PO-2.npy",
    "D3RLPY-PO-MP2-P0_20000.npy",
    "D3RLPY-PO-MP2-P1_20000.npy",
    "D3RLPY-PO-MP2-P2_20000.npy",
    "D3RLPY-PO-MP2-P3_20000.npy",
    "D3RLPY-PO-MP2-P4_20000.npy",
    "D3RLPY-PO-MP2-P0_100000.npy",
    "D3RLPY-PO-MP2-P1_100000.npy",
    "D3RLPY-PO-MP2-P2_100000.npy",
    "D3RLPY-PO-MP2-P3_100000.npy",
    "D3RLPY-PO-MP2-P4_100000.npy",
    "D3RLPY-PO-PAP2-P0_20000.npy",
    "D3RLPY-PO-PAP2-P1_20000.npy",
    "D3RLPY-PO-PAP2-P2_20000.npy",
    "D3RLPY-PO-PAP2-P3_20000.npy",
    "D3RLPY-PO-PAP2-P4_20000.npy",
    "D3RLPY-PO-PAP2-P0_100000.npy",
    "D3RLPY-PO-PAP2-P1_100000.npy",
    "D3RLPY-PO-PAP2-P2_100000.npy",
    "D3RLPY-PO-PAP2-P3_100000.npy",
    "D3RLPY-PO-PAP2-P4_100000.npy",
    "RAND-1-PO-2.npy",
    "RAND-2-PO-2.npy",
    "RAND-3-PO-2.npy",
    "RAND-4-PO-2.npy",
    "RAND-5-PO-2.npy",
    "RAND-6-PO-2.npy",
    "RAND-7-PO-2.npy",
    "RAND-8-PO-2.npy",
    "RAND-9-PO-2.npy",
    "RAND-10-PO-2.npy",
    "RAND-D3RLPY-PO-MP2-P0-1_100000.npy",
    "RAND-D3RLPY-PO-MP2-P1-1_100000.npy",
    "RAND-D3RLPY-PO-MP2-P2-1_100000.npy",
    "RAND-D3RLPY-PO-MP2-P3-1_100000.npy",
    "RAND-D3RLPY-PO-MP2-P4-1_100000.npy",
    "RAND-D3RLPY-PO-PAP2-P0-1_100000.npy",
    "RAND-D3RLPY-PO-PAP2-P1-1_100000.npy",
    "RAND-D3RLPY-PO-PAP2-P2-1_100000.npy",
    "RAND-D3RLPY-PO-PAP2-P3-1_100000.npy",
    "RAND-D3RLPY-PO-PAP2-P4-1_100000.npy",
]

if __name__ == "__main__":
    experiment = sys.argv[1]
    score_model(experiment, DATA_PATHS)
