import sys

from dogo.wasserstein.wasserstein_distance import wasserstein_distances

MIXED_3_PO_1_DATASETS = [
    # "MIXED-3-PO-2",
    "D3RLPY-PO-MP2-P0_100000",
    "D3RLPY-PO-MP2-P1_100000",
    "D3RLPY-PO-MP2-P2_100000",
    "D3RLPY-PO-MP2-P3_100000",
    "D3RLPY-PO-MP2-P4_100000",
    "D3RLPY-PO-PAP2-P0_100000",
    "D3RLPY-PO-PAP2-P1_100000",
    "D3RLPY-PO-PAP2-P2_100000",
    "D3RLPY-PO-PAP2-P3_100000",
    "D3RLPY-PO-PAP2-P4_100000",
    "RAND-1-PO-2",
    "RAND-2-PO-2",
    "RAND-3-PO-2",
    "RAND-4-PO-2",
    "RAND-5-PO-2",
    # "RAND-6-PO-2",
    # "RAND-7-PO-2",
    # "RAND-8-PO-2",
    # "RAND-9-PO-2",
    # "RAND-10-PO-2",
    "RAND-D3RLPY-PO-MP2-P0-1_100000",
    "RAND-D3RLPY-PO-MP2-P1-1_100000",
    "RAND-D3RLPY-PO-MP2-P2-1_100000",
    "RAND-D3RLPY-PO-MP2-P3-1_100000",
    "RAND-D3RLPY-PO-MP2-P4-1_100000",
    "RAND-D3RLPY-PO-PAP2-P0-1_100000",
    "RAND-D3RLPY-PO-PAP2-P1-1_100000",
    "RAND-D3RLPY-PO-PAP2-P2-1_100000",
    "RAND-D3RLPY-PO-PAP2-P3-1_100000",
    "RAND-D3RLPY-PO-PAP2-P4-1_100000",
]

if __name__ == '__main__':
    exp = sys.argv[1]
    wasserstein_distances(exp, MIXED_3_PO_1_DATASETS)
