"""Shared configuration for MVCD."""

# --- paper view definition ----------------------------------------------
# Three image views plus two complementary text views for each of the six
# corpus-frequency terms. The construction does not use GT factor names/counts.
TEXT_KEYWORD_METHOD = "frequency"
TEXT_KEYWORD_K = 6


def views(dataset):
    if dataset not in ("MAG", "Lung"):
        raise ValueError(dataset)
    return ["img_color", "img_hpf", "img_silhouette", *(
        f"txt_iso_{index:02d}_{side}"
        for index in range(TEXT_KEYWORD_K) for side in ("a", "b"))]


def n_views(dataset):
    return len(views(dataset))


def subsets(dataset):
    if dataset not in ("MAG", "Lung"):
        raise ValueError(dataset)
    return [(0, 1), (0, 2), (1, 2), *(
        (3 + 2 * index, 4 + 2 * index) for index in range(TEXT_KEYWORD_K))]


def image_views(dataset):
    return [i for i, v in enumerate(views(dataset)) if v.startswith("img_")]


def text_views(dataset):
    return [i for i, v in enumerate(views(dataset)) if v.startswith("txt_")]


# --- MultiView / encoder hyperparameters (short-term defaults) ------------
ENCODING_SIZE = 32          # per-view adapter output dim
SELECTION = "gumbel_softmax"

# Contrastive loss for content alignment across a subset's views. Both estimate
# the same objective (align content + entropy/redundancy reg); the MultiView
# paper uses BOTH (InfoNCE and BarlowTwins, the latter for multimodal), but the
# released code only ships InfoNCE, so "barlow" is a paper-based extension.
#   "infonce" -> multi-view InfoNCE with -L2^2 similarity (reproduction)
#   "barlow"  -> BarlowTwins: cross-correlation of view content -> identity
LOSS = "infonce"
# LOSS_SIM: similarity inside InfoNCE. "l2" -> -||c-c'||^2 (reproduction, default);
#   "dot" -> <c,c'> inner product. ABLATION AXIS (independent of NAMING_USE_LOSS_SIM).
LOSS_SIM = "l2"
# Normalize each subset's loss by its content size. NOTE: empirically this HURT
# stability (flattens larger-k softmax) -- off by default; kept as a knob.
LOSS_DIM_NORMALIZE = False
BARLOW_LAMBDA = 1.0         # off-diagonal (redundancy) weight in BarlowTwins

# Which CLIP feature to extract, per modality. The final projection maps into the
# shared image-text contrastive space where holistic/sentiment signal dominates
# (diagnosed bottleneck), so we default to pre-projection features:
#   "final"       -> CLIP's projected embedding (512) -- holistic baseline
#   "preproj"     -> pre-projection pooled token (image CLS 768 / text EOT 512)
#   "tokens_mean" -> mean over valid tokens before projection (image 768 / text 512)
# Text uses tokens_mean (spreads signal across words so masked-category words
# matter); image uses preproj (keeps 768-d detail before the contrastive squeeze).
IMAGE_FEATURE = "preproj"
TEXT_FEATURE = "tokens_mean"

# --- content size (block dimensionality), PER-SUBSET like the MultiView authors -
# The paper/authors assign each subset its own content size = the number of latents
# its views share (Eq 2.2). The released paper configuration learns this size.
#   CONTENT_SIZE_MODE:
#     "per_subset" (default) -> _CONTENT_SIZES below (reproduces authors' per-subset k)
#     "uniform"              -> CONTENT_SIZE_UNIFORM for every subset
#     "variable"             -> learn the size (Thm 3.8 L0 gate) [phase 2; caps at UNIFORM]
# DEFAULT = "uniform": an 8-GPU sweep showed uniform k=4 passes 11/11 subsets on
# both datasets (vs per_subset 2/11 MAG) -- larger k is far more seed-stable on
# small noisy data, and decompose then recovers the real factors. "per_subset"
# (author-faithful, sizes = design intersection) stays available but regresses here.
CONTENT_SIZE_MODE = "uniform"
CONTENT_SIZE_UNIFORM = 4
# "variable" mode: learn k per subset via per-dim gates + an L0 sparsity penalty
# (Thm 3.8 / Def 3.7). CAVEAT (measured): flexible InfoNCE keeps reducing loss as
# more dims are added (no clean stopping point), so the penalty sets a roughly
# global k rather than recovering each subset's true size; alpha ~0.005 lands k~6-9.
VARIABLE_L0_ALPHA = 0.005
# validity trains at the SAME content size as finalize by DEFAULT (None -> match
# CONTENT_SIZE_MODE). Set to a mode string ("uniform"/"per_subset") to decouple
# validity's training dim from finalize's (e.g. validate at uniform, finalize at
# per_subset). Kept as an option; matched is recommended.
VALIDITY_CONTENT_SIZE_MODE = None

_CONTENT_SIZES = {
    dataset: {subset: (2 if subset[0] < 3 else 1)
              for subset in subsets(dataset)}
    for dataset in ("MAG", "Lung")
}

# Decomposition is an independent on/off knob (NOT tied to content size):
#   True  -> split each >=2-D block into 1-D axes via sparse local difference
#   False -> keep the multi-D block as-is (naming/pairs use multi-D L2 distance)
# NODE-FIRST DEFAULT OFF: decompose is not proposed by either source paper; the
# node-focused phase keeps subsets un-split. Turn on only for the edge phase.
DECOMPOSE_ENABLED = False

# --- Node-first phase toggles (see docs/개선작업_이해_및_계획.md) -----------
# RANK_NORMALIZE: rank-normalize each block column before downstream. OFF -> use
#   the raw projection P^{(b)} directly (naming pairs then use raw-L2). Default OFF.
RANK_NORMALIZE = False
# DEDUP_ENABLED: merge near-collinear atoms (|corr|>DEDUP_THRESHOLD) into one
#   factor. OFF -> keep every atom as its own factor (no merge). Default OFF.
DEDUP_ENABLED = False


def content_size_for(dataset, subset, mode=None):
    """Content size (block dim) for ONE subset under the current mode."""
    mode = mode or CONTENT_SIZE_MODE
    if mode in ("uniform", "variable"):           # variable caps at UNIFORM (phase 2)
        return CONTENT_SIZE_UNIFORM
    if mode == "per_subset":
        return _CONTENT_SIZES[dataset].get(tuple(subset), CONTENT_SIZE_UNIFORM)
    raise ValueError(f"unknown CONTENT_SIZE_MODE {mode}")


def content_sizes(dataset, subsets_=None, mode=None):
    """Per-subset content sizes, aligned with `subsets_` (default: all subsets)."""
    subs = subsets_ if subsets_ is not None else subsets(dataset)
    return [content_size_for(dataset, s, mode) for s in subs]

# Sparse local difference decomposition knobs.
DECOMP_KNN = 10
DECOMP_MAX_AXES = None
DECOMP_MIN_SUPPORT = 0.10
DECOMP_TIGHTNESS = 0.6

# --- Validity gate (block pass/fail) --------------------------------------
# The gate is 1-D (see run_validity): a subset passes if its shared block is
# permutation-significant in >= PERM_FOLD_FRACTION folds AND seed-stable.
# VALIDITY_GATE_ENABLED off -> skip seed/fold perm+stability entirely and pass
# ALL subsets straight to finalize (which retrains on full data). ABLATION AXIS.
VALIDITY_GATE_ENABLED = True
PERM_P_CUT = 0.05            # permutation-test significance
PERM_FOLD_FRACTION = 3       # need p<PERM_P_CUT in >= this many of N_FOLDS
STABILITY_CUT = 0.60         # min mean |rho| of block across seeds
N_PERM = 1000                # permutations per test
N_SEEDS = 3                  # seeds for stability
N_FOLDS = 5                  # CV folds
RUN_SEED = 0                 # paper experiments use outer seeds {0,1,2,3,4}
VALIDITY_STEPS = 3000        # training steps in validity (1D)
FINALIZE_STEPS = 3000        # training steps in finalize (full-data retrain)
DEDUP_THRESHOLD = 0.9        # merge atoms with |corr| > this before LLM naming

# --- Naming (LLM contrastive pair selection) ------------------------------
TOP_N = 5                    # pairs shown to the LLM per block (MLLM-CD K=5); ABLATION {5,10}
# One score formula is used: d_target - lambda * sum(d_other).
# True -> lambda=1/(M-1); False -> lambda=1. No separate numeric lambda knob.
LAMBDA_AVERAGE = True
# NAMING_USE_LOSS_SIM: pick naming pairs with the SAME similarity the loss uses
#   (LOSS_SIM below) instead of the fixed L2 distance. ABLATION AXIS (independent
#   of LOSS_SIM -> 4 combos). off -> naming uses L2 distance (current).
NAMING_USE_LOSS_SIM = False
# Name blocks sequentially, each told to stay DISTINCT from already-named factors
# (and, for the LLM merge, merge only when clearly overlapping) -- reduces the
# LLM's tendency to over-merge value-distinct blocks into one umbrella factor.
NAMING_DISTINCT_FIRST = True

# --- Quantification (per-sample factor values fed to FCI) ------------------
# "llm"       -> LLM 1/0/-1 annotation (reproduction); factors = LLM-merged names.
# "multiview" -> the math block representation (block_rank01) directly; factors =
#                the deduped blocks (no LLM merge), LLM only names them.
QUANT = "llm"

# NODES_ONLY (node-first phase): stop after factor naming, export a node-only
# graph (factors + Y, no edges), skip quantification/FCI/counterfactual. Metrics
# then report NP/NR/NF only. Also settable via `main_mv --nodes-only`.
NODES_ONLY = False

# Only for QUANT="multiview": collapse the continuous block_rank01 to tercile
# +1/0/-1 (bottom third / middle / top third) before FCI, matching the LLM
# annotation's 3-level distribution. Denoises the uniform-rank middle mass and
# restores a neutral (0) category -> measurably better edge recovery on MAG.
MULTIVIEW_DISCRETIZE = False

# Data locations (relative to project root).
DATA_ROOT = "mvext_data"


# --- external override file -----------------------------------------------
# Everything above is the DEFAULT. To sweep hyperparameters, point the env var
# MVEXT_CONFIG at a small python file in configs/ that reassigns any of the names
# above (e.g. `STABILITY_CUT = 0.5`). run_mvext.sh sets MVEXT_CONFIG from its
# CONFIG variable. Only top-level names are copied over; unknown names are added.
def _apply_overrides():
    import os
    path = os.environ.get("MVEXT_CONFIG")
    if not path:
        return
    if not os.path.isfile(path):
        raise FileNotFoundError(f"MVEXT_CONFIG file not found: {path}")
    ns = {}
    with open(path) as f:
        exec(compile(f.read(), path, "exec"), ns)
    g = globals()
    applied = {}
    for k, val in ns.items():
        if k.startswith("_") or callable(val) or k in ("os",):
            continue
        g[k] = val
        applied[k] = val
    print(f"[config] overrides from {path}: {applied}")


_apply_overrides()
