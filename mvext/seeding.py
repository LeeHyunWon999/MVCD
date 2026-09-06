"""Outer-seed handling shared by run4 and the original baselines."""
import os
import random

import numpy as np
import torch


_STRIDE = 10_000


def seed_everything(run_seed):
    run_seed = int(run_seed)
    os.environ["MVEXT_RUN_SEED"] = str(run_seed)
    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)
    return run_seed


def validity_seed(run_seed, inner_seed):
    return int(run_seed) * _STRIDE + int(inner_seed)


def permutation_seed(run_seed, fold):
    return int(run_seed) * _STRIDE + 1_000 + int(fold)


def finalize_seed(run_seed):
    return int(run_seed) * _STRIDE + 12_345
