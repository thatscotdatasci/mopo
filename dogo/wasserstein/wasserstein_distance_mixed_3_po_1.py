import sys

from dogo.wasserstein.wasserstein_distance import wasserstein_distances

MIXED_3_PO_2_DATASETS = [
    # "MIXED-3-PO-1",
    "D3RLPY-PO-MP1-P0_100000",
    "D3RLPY-PO-MP1-P1_100000",
    "D3RLPY-PO-MP1-P2_100000",
    "D3RLPY-PO-MP1-P3_100000",
    "D3RLPY-PO-MP1-P4_100000",
    "D3RLPY-PO-PAP1-P0_100000",
    "D3RLPY-PO-PAP1-P1_100000",
    "D3RLPY-PO-PAP1-P2_100000",
    "D3RLPY-PO-PAP1-P3_100000",
    "D3RLPY-PO-PAP1-P4_100000",
    "RAND-1-PO-1",
    "RAND-2-PO-1",
    "RAND-3-PO-1",
    "RAND-4-PO-1",
    "RAND-5-PO-1",
    # "RAND-6-PO-1",
    # "RAND-7-PO-1",
    # "RAND-8-PO-1",
    # "RAND-9-PO-1",
    # "RAND-10-PO-1",
    "RAND-D3RLPY-PO-MP1-P0-1_100000",
    "RAND-D3RLPY-PO-MP1-P1-1_100000",
    "RAND-D3RLPY-PO-MP1-P2-1_100000",
    "RAND-D3RLPY-PO-MP1-P3-1_100000",
    "RAND-D3RLPY-PO-MP1-P4-1_100000",
    "RAND-D3RLPY-PO-PAP1-P0-1_100000",
    "RAND-D3RLPY-PO-PAP1-P1-1_100000",
    "RAND-D3RLPY-PO-PAP1-P2-1_100000",
    "RAND-D3RLPY-PO-PAP1-P3-1_100000",
    "RAND-D3RLPY-PO-PAP1-P4-1_100000",
]

if __name__ == '__main__':
    exp = sys.argv[1]
    wasserstein_distances(exp, MIXED_3_PO_2_DATASETS)
