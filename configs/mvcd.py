"""Paper configuration for MVCD with corpus-frequency text views."""

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
NODES_ONLY = False
TEXT_KEYWORD_METHOD = "frequency"
TEXT_KEYWORD_K = 6
MULTIVIEW_DISCRETIZE = False
DATA_ROOT = "mvext_data"
