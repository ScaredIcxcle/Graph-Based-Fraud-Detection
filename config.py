"""
config.py
Shared settings for the whole pipeline. Edit paths here if your CSV files
are named/located differently.
"""

import os

# ---- INPUT FILES (edit these paths to match your downloaded dataset) ----
TRAIN_CSV = "fraudTrain.csv"
TEST_CSV = "fraudTest.csv"

# ---- OUTPUT DIRECTORY ----
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- PATTERN TAGGING THRESHOLDS (percentile-based, computed once per split) ----
FAN_IN_PERCENTILE = 95
FAN_OUT_PERCENTILE = 95
CHAIN_TIME_DELTA_MINUTES = 10
CHAIN_MIN_LENGTH = 3
DENSE_COMMUNITY_PERCENTILE = 90
LOUVAIN_RESOLUTION = 10.0  # higher = more, smaller communities (default 1.0 was collapsing to ~3 giant blobs)

# ---- COMMUNITY FRAUD RATE SMOOTHING ----
# Bayesian smoothing strength: higher = small communities trust the global
# average more; lower = small communities trust their own (noisier) rate more.
FRAUD_RATE_PRIOR_WEIGHT = 50

# ---- MODEL SETTINGS ----
RANDOM_SEED = 42
N_FOLDS = 5
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "aucpr",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}