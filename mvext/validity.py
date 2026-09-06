"""Module 4 - block validity (5-fold x 3-seed) + final retrain.

Filters the 8 candidate subsets down to those whose shared block is actually
supported, then retrains once on the full data with a fresh seed to produce the
final block model.

Two tests (see MLLM-CD-additional-info/block factor 유효성 판정 기술.txt):
  - Permutation (per fold, on seed 0): is the matched cross-view residual
    significantly smaller than when one view's sample order is shuffled? Uses
    per-view projections WITHOUT averaging. Pass a fold if p < 0.05.
  - Seed stability (per fold, across 3 seeds): do seeds agree on the sample
    ordering of the (view-averaged) block value? 1-D -> mean |Spearman rho|.
    (2-D+ -> distance-matrix Spearman; implemented but short-term uses 1-D.)

Aggregation: a subset is kept if permutation passes in >= 3/5 folds AND the
fold-mean seed stability >= 0.6.

Usage:
    python -m mvext.validity --dataset MAG
"""
import argparse
import itertools

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

from mvext import config as C
from mvext.encoders import load_cache
from mvext.multiview import train_multiview, extract_block_projections, average_views
from mvext.seeding import (finalize_seed, permutation_seed, seed_everything,
                           validity_seed)

# All gate hyperparameters live in config.py (single source of truth).
PERM_P_CUT = C.PERM_P_CUT
PERM_FOLD_FRACTION = C.PERM_FOLD_FRACTION
STABILITY_CUT = C.STABILITY_CUT
N_PERM = C.N_PERM
N_SEEDS = C.N_SEEDS
N_FOLDS = C.N_FOLDS


# --- statistics -----------------------------------------------------------


def _subset_residual(view_projs, perm_view=None, perm=None):
    """Mean over samples of the mean pairwise L2^2 residual across views.
    view_projs: dict view_idx -> (n, d). If perm given, apply it to perm_view."""
    views = list(view_projs.keys())
    mats = {k: view_projs[k] for k in views}
    if perm is not None:
        mats = dict(mats)
        mats[perm_view] = mats[perm_view][perm]
    total = 0.0
    npair = 0
    for a, b in itertools.combinations(views, 2):
        total = total + ((mats[a] - mats[b]) ** 2).sum(-1)   # (n,)
        npair += 1
    return float((total / npair).mean())


def permutation_pvalue(view_projs, n_perm=N_PERM, rng=None):
    """Empirical p = P(R_perm <= R_matched). Small p => matched significantly
    smaller => block supported. One view (the last) is shuffled per round."""
    rng = rng or np.random.default_rng(0)
    views = list(view_projs.keys())
    perm_view = views[-1]
    n = view_projs[perm_view].shape[0]
    r_matched = _subset_residual(view_projs)
    le = 0
    for _ in range(n_perm):
        p = torch.from_numpy(rng.permutation(n))
        if _subset_residual(view_projs, perm_view, p) <= r_matched:
            le += 1
    return (le + 1) / (n_perm + 1), r_matched


def seed_stability(block_by_seed):
    """block_by_seed: list over seeds of (n, d) view-averaged block values.
    1-D -> mean |Spearman| over seed pairs. 2-D+ -> distance-matrix Spearman."""
    d = block_by_seed[0].shape[1]
    vals = []
    for s1, s2 in itertools.combinations(range(len(block_by_seed)), 2):
        a, b = block_by_seed[s1].numpy(), block_by_seed[s2].numpy()
        if d == 1:
            rho, _ = spearmanr(a[:, 0], b[:, 0])
            vals.append(abs(rho))
        else:
            da = np.linalg.norm(a[:, None] - a[None], axis=-1)
            db = np.linalg.norm(b[:, None] - b[None], axis=-1)
            iu = np.triu_indices(len(a), k=1)
            rho, _ = spearmanr(da[iu], db[iu])
            vals.append(rho)
    return float(np.mean(vals)) if vals else 0.0


# --- orchestration --------------------------------------------------------


def run_validity(dataset, steps=C.VALIDITY_STEPS, n_perm=N_PERM, content_sizes=None,
                 verbose=True, run_seed=C.RUN_SEED):
    cache = load_cache(dataset)
    feats = cache["features"]                    # (N, n_views, D)
    N = feats.shape[0]
    kf = KFold(n_splits=N_FOLDS, shuffle=False)   # sample<->view alignment fixed

    subsets = [tuple(s) for s in C.subsets(dataset)]
    # Gate OFF -> skip the perm/stability folds and keep ALL subsets (finalize
    # retrains on full data anyway). keep=True for every subset.
    if not getattr(C, "VALIDITY_GATE_ENABLED", True):
        if verbose:
            print(f"[{dataset}] validity gate OFF -> keep all {len(subsets)} subsets")
        return {s: {"perm_folds_passed": None, "stability": None, "keep": True}
                for s in subsets}
    # Validity trains at the SAME per-subset content sizes used downstream (no
    # separate 1-D proxy): seed_stability adapts (1-D rank vs multi-D distance).
    # matches finalize by default (VALIDITY_CONTENT_SIZE_MODE=None); override to decouple
    vmode = C.VALIDITY_CONTENT_SIZE_MODE or C.CONTENT_SIZE_MODE
    csizes = content_sizes or C.content_sizes(dataset, mode=vmode)
    variable = (vmode == "variable")
    perm_pass = {s: 0 for s in subsets}          # count of folds with p<0.05 (seed 0)
    stab_scores = {s: [] for s in subsets}       # per-fold mean |rho|

    for fold, (tr, te) in enumerate(kf.split(np.arange(N))):
        tr_feats, te_feats = feats[tr], feats[te]
        proj_by_seed = []
        for inner_seed in range(N_SEEDS):
            model, _ = train_multiview(tr_feats, subsets=subsets, content_sizes=csizes,
                                       variable=variable, steps=steps,
                                       seed=validity_seed(run_seed, inner_seed), verbose=False)
            proj_by_seed.append(extract_block_projections(model, te_feats))

        for s in subsets:
            # permutation on seed 0 (per-view, no averaging)
            p, _ = permutation_pvalue(proj_by_seed[0][s], n_perm=n_perm,
                                      rng=np.random.default_rng(permutation_seed(run_seed, fold)))
            if p < PERM_P_CUT:
                perm_pass[s] += 1
            # seed stability across all seeds (view-averaged)
            blocks = [average_views(proj_by_seed[sd][s]) for sd in range(N_SEEDS)]
            stab_scores[s].append(seed_stability(blocks))
        if verbose:
            print(f"[{dataset}] fold {fold} done")

    verdict = {}
    for s in subsets:
        stab = float(np.mean(stab_scores[s]))
        keep = (perm_pass[s] >= PERM_FOLD_FRACTION) and (stab >= STABILITY_CUT)
        verdict[s] = {"perm_folds_passed": perm_pass[s], "stability": round(stab, 3), "keep": keep}
    return verdict


def final_retrain(dataset, passed_subsets, seed=None, steps=C.FINALIZE_STEPS, standardize=False,
                  content_sizes=None):
    """Retrain on FULL data with only the validated subsets and a fresh seed."""
    seed = finalize_seed(C.RUN_SEED) if seed is None else seed
    cache = load_cache(dataset)
    csizes = content_sizes or C.content_sizes(dataset, subsets_=passed_subsets)
    model, info = train_multiview(
        cache["features"], subsets=passed_subsets, content_sizes=csizes,
        variable=(C.CONTENT_SIZE_MODE == "variable"),
        steps=steps, seed=seed, standardize=standardize, verbose=True)
    return model, info


def rank_normalize(x):
    """Rank-normalize a (N,1) column to [0,1]. Block-identifiability preserves
    only the ordering (h is unknown but rank-preserving in 1-D), so we quantify
    by rank, not raw magnitude."""
    order = x[:, 0].argsort()
    ranks = torch.empty_like(x[:, 0])
    ranks[order] = torch.arange(len(x), dtype=x.dtype)
    return (ranks / max(len(x) - 1, 1)).unsqueeze(1)


def rank_normalize_cols(v):
    """Rank-normalize EACH column of (N,d) to [0,1]. For d=1 this equals
    rank_normalize; for d>1 it makes multi-D block values scale-comparable so
    pair-selection can use L2 distance uniformly (decompose-off case)."""
    return torch.cat([rank_normalize(v[:, j:j + 1]) for j in range(v.shape[1])], dim=1)


def reduce_axis(v):
    """Weight vector (d,) that collapses a rank-normalized block value (N,d) to a
    1-D summary via `v @ axis`. d==1 -> [1]. d>1 -> the first PRINCIPAL COMPONENT
    (max-variance direction), sign-oriented so the summary agrees with the plain
    column mean (keeps factor polarity stable). This replaces the earlier column
    mean (axis = 1/d): PCA-1 preserves more of the multi-D block's signal, which
    measurably improves edge recovery in the decompose-off (multi-D) case."""
    d = v.shape[1]
    if d == 1:
        return torch.ones(1, dtype=v.dtype)
    x = v.numpy()
    xc = x - x.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(xc, full_matrices=False)
    axis = Vt[0]                                      # unit-norm PC1 direction (d,)
    if np.corrcoef(x @ axis, x.mean(1))[0, 1] < 0:    # orient toward the mean
        axis = -axis
    return torch.tensor(axis, dtype=v.dtype)


def dedup_blocks(block_rank01, subsets, stability=None, threshold=0.9):
    """Merge near-duplicate blocks (|corr| > threshold) via connected components;
    keep one representative per component (highest seed-stability, else first).

    Redundant image blocks would be near-collinear and break FCI faithfulness,
    so we collapse each cluster to a single factor before downstream use.
    Returns (kept_indices sorted, clusters list-of-index-lists)."""
    import numpy as np
    v = block_rank01.numpy()
    n = v.shape[1]
    corr = np.abs(np.corrcoef(v.T)) if n > 1 else np.array([[1.0]])

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if corr[i, j] > threshold:
                parent[find(i)] = find(j)

    comps = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    clusters = list(comps.values())

    kept = []
    for members in clusters:
        if stability is not None:
            rep = max(members, key=lambda i: stability.get(tuple(subsets[i]), 0.0))
        else:
            rep = members[0]
        kept.append(rep)
    return sorted(kept), clusters


def finalize(dataset, passed_subsets, verdict=None, seed=None, steps=C.FINALIZE_STEPS,
             standardize=False, dedup_threshold=C.DEDUP_THRESHOLD, decompose=None,
             root=None, run_seed=C.RUN_SEED):
    """Retrain on full data with the validated subsets and persist everything
    module 5 needs. Blocks are trained at their per-subset content size. If
    `decompose` (default C.DECOMPOSE_ENABLED), each >=2-D block is split into 1-D
    factor axes (sparse local difference) before dedup, so a subset can yield
    several 1-D factors; otherwise the multi-D block is kept as ONE factor whose
    per-sample value is multi-dimensional (naming/pairs then use L2 distance)."""
    import os
    from mvext.decompose import decompose_block
    root = root or C.DATA_ROOT
    seed = finalize_seed(run_seed) if seed is None else seed
    decompose = C.DECOMPOSE_ENABLED if decompose is None else decompose
    cache = load_cache(dataset)
    full_subsets = [tuple(s) for s in passed_subsets]
    if not full_subsets:
        empty = torch.empty((cache["features"].shape[0], 0))
        out = {
            "dataset": dataset,
            "passed_subsets": [],
            "block_subsets": [],
            "dedup_clusters": [],
            "dedup_threshold": dedup_threshold,
            "verdict": verdict,
            "model_state": None,
            "model_config": {
                "subsets": [], "n_views": cache["features"].shape[1],
                "in_dim": cache.get("dim", cache["features"].shape[-1]),
                "enc": C.ENCODING_SIZE, "content_sizes": [],
                "variable": C.CONTENT_SIZE_MODE == "variable",
            },
            "selected_dims": [],
            "feature_stats": None,
            "block_proj_per_view": {},
            "block_values": [],
            "block_axes": [],
            "block_raw": [],
            "block_reduce_axes": [],
            "block_value": empty,
            "block_rank01": empty,
            "block_value_full": empty,
            "block_rank01_full": empty,
            "decompose": decompose,
            "score": cache["score"],
            "seed": seed,
            "standardize": standardize,
            "empty_reason": "no_valid_subsets",
        }
        os.makedirs(os.path.join(root, dataset), exist_ok=True)
        path = os.path.join(root, dataset, "blocks.pt")
        torch.save(out, path)
        print(f"[{dataset}] 0 passed -> target-only artifact -> {path}")
        return out

    model, info = final_retrain(dataset, passed_subsets, seed, steps, standardize)
    stats = info["feature_stats"]
    proj = extract_block_projections(model, cache["features"], stats=stats)

    # Build "atoms". decompose: split each >=2-D block into 1-D axes (one atom
    # per axis). else: one atom per subset (value may be multi-D). Each atom's
    # value is rank-normalized per column so blocks are scale-comparable.
    # Per atom we also keep the projection AXIS (subset block -> atom, for freezing
    # new samples) and the RAW value (to rank future samples against a fixed map).
    # RANK_NORMALIZE off (node-first default) -> block values ARE the raw projection
    # P^{(b)}; naming pairs then use raw-L2. on -> rank-normalize each column.
    def _val(x):
        return rank_normalize_cols(x) if getattr(C, "RANK_NORMALIZE", True) else x
    atom_subsets, atom_vals, atom_axes, atom_raw = [], [], [], []
    for s in full_subsets:
        P = average_views(proj[s])                       # (N, k_s)
        if decompose and P.shape[1] > 1:
            vals, axes = decompose_block(P.numpy())      # vals (N,n_axes), axes (n_axes,k)
            for a in range(vals.shape[1]):
                raw = torch.tensor(vals[:, a:a + 1], dtype=torch.float)
                atom_subsets.append(s)
                atom_vals.append(_val(raw))
                atom_axes.append(torch.tensor(axes[a], dtype=torch.float))   # (k,) proj axis
                atom_raw.append(raw)                     # (N,1) raw for ranking
        else:
            atom_subsets.append(s)
            atom_vals.append(_val(P))                    # (N, k_s), rank per col (or raw)
            atom_axes.append(None)                       # multi-D: no projection (use P)
            atom_raw.append(P)                           # (N, k_s) raw

    # 1-D summary per atom via PCA-1 over its rank columns (was: column mean).
    # For 1-D atoms this is identity; for multi-D (decompose-off) it takes the
    # dominant axis instead of averaging, preserving more signal for FCI.
    atom_reduce = [reduce_axis(v) for v in atom_vals]      # per atom: (d_i,) weights
    summaries = torch.cat([(v * ax).sum(1, keepdim=True)
                           for v, ax in zip(atom_vals, atom_reduce)], dim=1)

    # collapse near-duplicate atoms before downstream (FCI collinearity).
    # DEDUP_ENABLED off (node-first default) -> keep every atom as its own factor.
    if getattr(C, "DEDUP_ENABLED", True):
        stab = {tuple(k): v["stability"] for k, v in (verdict or {}).items()
                if v["stability"] is not None}
        kept, clusters = dedup_blocks(summaries, atom_subsets, stability=stab or None,
                                      threshold=dedup_threshold)
    else:
        kept = list(range(len(atom_subsets)))            # no merge: all atoms kept
        clusters = [[i] for i in kept]
    block_subsets = [atom_subsets[i] for i in kept]
    block_values = [atom_vals[i] for i in kept]          # list of (N, d_i) rank-normed
    block_axes = [atom_axes[i] for i in kept]            # per factor: (k,) axis or None
    block_raw = [atom_raw[i] for i in kept]              # per factor: raw values (fixed map)
    block_reduce_axes = [atom_reduce[i] for i in kept]   # per factor: (d_i,) 1-D reduce weights
    block_rank01 = summaries[:, kept]                    # (N, n_kept) 1-D summaries
    block_value = block_rank01                           # back-compat alias
    value_full = summaries
    rank_full = summaries
    cluster_map = [[atom_subsets[i] for i in c] for c in clusters]

    out = {
        "dataset": dataset,
        "passed_subsets": full_subsets,          # all validity-passed
        "block_subsets": block_subsets,          # deduped -> downstream factors
        "dedup_clusters": cluster_map,
        "dedup_threshold": dedup_threshold,
        "verdict": verdict,
        "model_state": model.state_dict(),
        "model_config": {
            "subsets": full_subsets, "n_views": cache["features"].shape[1],
            "in_dim": cache["dim"], "enc": C.ENCODING_SIZE,
            "content_sizes": model.content_sizes, "variable": model.variable,
        },
        "selected_dims": info["selected_dims"],
        "feature_stats": stats,
        "block_proj_per_view": {s: {k: proj[s][k] for k in proj[s]} for s in full_subsets},
        "block_values": block_values,            # list of (N, d_i) rank-normed per factor
        "block_axes": block_axes,                # per factor: (k,) decompose axis or None
        "block_raw": block_raw,                  # per factor: raw values (fixed rank map)
        "block_reduce_axes": block_reduce_axes,  # per factor: (d_i,) PCA-1 reduce weights

        "block_value": block_value,              # (N, n_kept) 1-D summaries (back-compat)
        "block_rank01": block_rank01,            # (N, n_kept) 1-D summaries
        "block_value_full": value_full,
        "block_rank01_full": rank_full,
        "decompose": decompose,
        "score": cache["score"],
        "seed": seed, "standardize": standardize,
    }
    os.makedirs(os.path.join(root, dataset), exist_ok=True)
    path = os.path.join(root, dataset, "blocks.pt")
    torch.save(out, path)
    print(f"[{dataset}] {len(full_subsets)} passed -> {len(block_subsets)} after dedup -> {path}")
    print(f"  clusters: {cluster_map}")
    print(f"  kept blocks: {block_subsets}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["MAG", "Lung", "both"], required=True)
    ap.add_argument("--steps", type=int, default=C.VALIDITY_STEPS)
    ap.add_argument("--finalize-steps", type=int, default=None)  # defaults to --steps
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--run-seed", type=int, default=C.RUN_SEED)
    ap.add_argument("--output-root", default=None,
                    help="write blocks here while reading the cache from DATA_ROOT")
    ap.add_argument("--no-finalize", action="store_true")
    args = ap.parse_args()
    run_seed = seed_everything(args.run_seed)
    for ds in (["MAG", "Lung"] if args.dataset == "both" else [args.dataset]):
        print(f"\n===== validity: {ds} =====")
        verdict = run_validity(ds, steps=args.steps, n_perm=args.n_perm,
                               run_seed=run_seed)
        print(f"\n{'subset':14s} {'perm(≥3/5)':>11s} {'stability':>10s} {'keep':>6s}")
        for s, v in verdict.items():
            perm = "—" if v["perm_folds_passed"] is None else str(v["perm_folds_passed"])
            stability = "—" if v["stability"] is None else f"{v['stability']:.3f}"
            print(f"{str(s):14s} {perm:>11s} {stability:>10s} {str(v['keep']):>6s}")
        kept = [s for s, v in verdict.items() if v["keep"]]
        print(f"\n통과 subset {len(kept)}/{len(verdict)}: {kept}")
        if not args.no_finalize:
            if kept:
                print(f"[{ds}] finalize (content_size={C.CONTENT_SIZE_MODE}, "
                      f"decompose={C.DECOMPOSE_ENABLED}, "
                      f"steps={args.finalize_steps or args.steps}) -> blocks.pt")
            else:
                print(f"[{ds}] no subset passed -> target-only blocks.pt")
            finalize(ds, kept, verdict=verdict, steps=args.finalize_steps or args.steps,
                     root=args.output_root, run_seed=run_seed)


if __name__ == "__main__":
    main()
