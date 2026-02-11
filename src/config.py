"""
This file contains constants and precomputed results which are used in other modules.
"""

DATASET_SPLITS = {
    "train": {"count": 69863, "full_name": "Training"},
    "val": {"count": 10000, "full_name": "Validation"},
}
BDD_LABELS_PREFIX = "bdd100k_labels_images_"
NETWORK_MOUNT = "/data"
CSV_DIR = "/code/src/analysis/csv/"
PRECOMPUTED_DIR = "/code/src/app/precomputed/"
CATEGORIES = {
    "traffic sign": 0,
    "traffic light": 1,
    "car": 2,
    "rider": 3,
    "motor": 4,
    "person": 5,
    "bus": 6,
    "truck": 7,
    "bike": 8,
    "train": 9,
}
