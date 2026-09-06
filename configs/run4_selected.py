"""Paper configuration for node discovery."""

CONTENT_SIZE_MODE = "variable"
LOSS = "infonce"
LOSS_SIM = "l2"
VALIDITY_GATE_ENABLED = False
DECOMPOSE_ENABLED = False
RANK_NORMALIZE = False
DEDUP_ENABLED = False
NAMING_USE_LOSS_SIM = False
TOP_N = 10
LAMBDA_AVERAGE = True
QUANT = "multiview"
NODES_ONLY = True
DATA_ROOT = "mvext_data"
