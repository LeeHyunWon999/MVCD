"""MultiView-CD driver (A1): the MLLM-CD loop with its CFD stage replaced by
MultiView block discovery + block-based naming (mvext modules 1-5). Originals
(main_MAG.py / main_Lung.py, utils.py) are left untouched for comparison.

iter1 (this file, module 5 finish):
    blocks.pt -> LLM factor naming -> LLM 1/0/-1 annotation -> KCI-FCI -> graph
    -> export graph JSON (eval-compatible) so metrics can be computed.
iter2 (module 6, separate) adds counterfactual refinement with adapter+block
frozen.

Usage:
    MODEL=gemini-2.5-flash-image API_KEY=... python -m mvext.main_mv --dataset MAG
"""
import argparse
import datetime
import json
import os
from copy import deepcopy

import pandas as pd
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.GraphUtils import GraphUtils

from utils import Helper
from gemini_utils import generate
from mvext import config as C
from mvext.naming import (load_blocks, discover_factors_from_blocks, annotate_llm,
                          resolve_top_n, select_block_pairs)
from mvext.seeding import seed_everything

try:
    from eval.graph_io import save_graph_from_causal_learn, save_graph_json, directed_graph_dict
except Exception:
    save_graph_from_causal_learn = save_graph_json = directed_graph_dict = None

TARGET = "score"


def _tercile(mat):
    """Collapse each column of a continuous matrix to -1/0/+1 by terciles (rank-
    based, so the split is robust to the marginal shape). Used for QUANT=
    'multiview' when C.MULTIVIEW_DISCRETIZE is on."""
    import numpy as np
    out = np.zeros(mat.shape, dtype=float)
    for j in range(mat.shape[1]):
        r = pd.Series(mat[:, j]).rank(method="average", pct=True).values
        col = np.zeros(len(r))
        col[r <= 1 / 3] = -1.0
        col[r >= 2 / 3] = 1.0
        out[:, j] = col
    return out


def load_meta(dataset):
    """Original per-dataset meta with ImagePath + Review, index 0..N-1 aligned to
    the manifest / blocks.pt order."""
    if dataset == "MAG":
        return pd.read_csv("MAG9.csv")
    if dataset == "Lung":
        ori = pd.read_csv("Lung.csv", index_col=0)
        meta = ori.iloc[:, :-3].copy()
        meta.index = range(len(meta))
        meta["Review"] = [f"{d} {h}".strip() for d, h in zip(ori["Demographics"], ori["History"])]
        meta["ImagePath"] = [f"./Lung/{i}.jpg" for i in ori.index]
        meta["score"] = ori["score"].values.astype(float)
        return meta
    raise ValueError(dataset)


def _unique_names(factors):
    """Ensure factor names are unique (they become DataFrame/graph node labels).
    Appends ' (2)', ' (3)' ... to collisions."""
    seen, out = {}, []
    for f in factors:
        nm = f["name"]
        if nm in seen:
            seen[nm] += 1
            nm = f"{nm} ({seen[f['name']]})"
        else:
            seen[nm] = 1
        out.append({**f, "name": nm})
    return out


def _quantify(dataset, helper, meta, factors, blocks, output_path, iter):
    """Per-sample factor values fed to FCI. QUANT='llm' -> LLM 1/0/-1 annotation;
    QUANT='multiview' -> the block representation (block_rank01), with factors
    aligned 1:1 to the deduped blocks."""
    csv = os.path.join(output_path, f"iter{iter}_annotation_results.csv")
    if C.QUANT == "multiview":
        r = blocks["block_rank01"].numpy()                 # (N, n_blocks) rank 0-1
        names = [f["name"] for f in factors]
        assert r.shape[1] == len(names), (
            f"multiview quant expects factors aligned with blocks "
            f"({r.shape[1]} blocks vs {len(names)} factors)")
        if getattr(C, "MULTIVIEW_DISCRETIZE", False):
            r = _tercile(r)                                # -> +1/0/-1 per column
        df = pd.DataFrame(r[:len(meta)], index=range(len(meta)), columns=names)
        df.to_csv(csv)
        return df
    ann = annotate_llm(helper, meta, factors, iter=iter)
    ann.to_csv(csv)
    return ann


def _discover_factors(dataset, helper, meta, blocks, output_path, top_n):
    """Factor discovery + naming (cached). Shared by iter1 and the nodes-only path."""
    fcache = os.path.join(output_path, "iter1_factors.json")
    if os.path.exists(fcache):
        factors = json.load(open(fcache))
        print(f"[{dataset}] loaded cached factors: {[f['name'] for f in factors]}")
    else:
        print(f"[{dataset}] block-based factor naming ({len(blocks['block_subsets'])} blocks, "
              f"quant={C.QUANT})")
        factors, _ = discover_factors_from_blocks(helper, meta, blocks, dataset,
                                                  top_n=top_n, iter=1, quant=C.QUANT)
        factors = _unique_names(factors)
        json.dump(factors, open(fcache, "w"), ensure_ascii=False, indent=2)
    print(f"[{dataset}] factors: {[f['name'] for f in factors]}")
    return factors


def run_nodes_only(dataset, helper, datetime_str, output_path, top_n=None):
    """Node-first path (B8): discover + name factors, export a NODE-ONLY graph
    ([Y] + factors, no edges). No quantification / FCI / counterfactual. Metrics
    then report NP/NR/NF only."""
    meta = load_meta(dataset)
    blocks = load_blocks(dataset)
    factors = _discover_factors(dataset, helper, meta, blocks, output_path,
                                resolve_top_n(top_n))
    names = [TARGET] + [f["name"] for f in factors]
    if save_graph_json is not None:
        model_tag = os.environ.get("MODEL", "unknown").replace("/", "_")
        gdir = os.path.join(output_path, model_tag)
        os.makedirs(gdir, exist_ok=True)
        gpath = os.path.join(gdir, f"graph_iter1_{datetime_str}.json")
        save_graph_json(directed_graph_dict(names, []), gpath)   # nodes only, no edges
        print(f"[{dataset}] node-only graph ({len(factors)} factors + Y) -> {gpath}")
    return factors


def run_iter1(dataset, helper, datetime_str, output_path, top_n=None):
    meta = load_meta(dataset)
    blocks = load_blocks(dataset)
    top_n = resolve_top_n(top_n)

    # Factor discovery is LLM-based and non-deterministic (temperature=1); cache
    # the result so re-runs are reproducible and skip the naming API calls. Pairs
    # are deterministic (block_rank01) so they are recomputed, not cached.
    pairs = {tuple(s): select_block_pairs(blocks["block_values"], b, top_n)
             for b, s in enumerate(blocks["block_subsets"])}
    factors = _discover_factors(dataset, helper, meta, blocks, output_path, top_n)

    ann = _quantify(dataset, helper, meta, factors, blocks, output_path, iter=1)
    data_interface = ann.copy()
    data_interface[TARGET] = meta["score"].values.astype(float)
    annotated_name = [TARGET] + [f["name"] for f in factors]
    values = data_interface[annotated_name].values.astype(float)

    g, _ = fci(values, alpha=0.05, independence_test_method="kci", verbose=False)
    print(f"[{dataset}] FCI done")

    pdy = GraphUtils.to_pydot(g, labels=annotated_name)
    print(pdy.to_string())
    if save_graph_from_causal_learn is not None:
        model_tag = os.environ.get("MODEL", "unknown").replace("/", "_")
        gdir = os.path.join(output_path, model_tag)
        os.makedirs(gdir, exist_ok=True)
        gpath = os.path.join(gdir, f"graph_iter1_{datetime_str}.json")
        save_graph_from_causal_learn(g, annotated_name, gpath)
        print(f"[{dataset}] graph JSON -> {gpath}")

    return factors, annotated_name, data_interface, pairs, g, pdy


def _export(save_g, labels, iter_idx, output_path, datetime_str):
    if save_graph_from_causal_learn is None:
        return
    model_tag = os.environ.get("MODEL", "unknown").replace("/", "_")
    gdir = os.path.join(output_path, model_tag)
    os.makedirs(gdir, exist_ok=True)
    gpath = os.path.join(gdir, f"graph_iter{iter_idx}_{datetime_str}.json")
    save_graph_from_causal_learn(save_g, labels, gpath)
    print(f"graph JSON -> {gpath}")


def _quantify_augmented(dataset, helper, meta_r, factors, blocks, output_path, iter, n_orig):
    """Per-sample factor values for the ORIGINAL + counterfactual samples (meta_r).
    QUANT='multiview': originals keep their block_rank01; the counterfactuals are
    frozen-projected (their views -> frozen CLIP+adapters+block axes -> fixed rank
    map). QUANT='llm': re-annotate everything with the iter1 factors."""
    csv = os.path.join(output_path, f"iter{iter}_annotation_results.csv")
    if C.QUANT == "multiview":
        from mvext.freeze import project_new_samples
        names = [f["name"] for f in factors]
        orig = blocks["block_rank01"].numpy()               # (n_orig, n_factors)
        rows = [orig]
        if len(meta_r) > n_orig:                              # counterfactuals appended
            cf = meta_r.iloc[n_orig:]
            cfvals = project_new_samples(dataset, cf["ImagePath"].tolist(),
                                         cf["Review"].tolist(), blocks).numpy()
            rows.append(cfvals)
        import numpy as np
        mat = np.vstack(rows)
        if getattr(C, "MULTIVIEW_DISCRETIZE", False):
            mat = _tercile(mat)                            # discretize over orig+CF
        df = pd.DataFrame(mat, index=range(len(meta_r)), columns=names)
        df.to_csv(csv)
        return df
    ann = annotate_llm(helper, meta_r, factors, iter=iter)
    ann.to_csv(csv)
    return ann


def refine_mv(dataset, helper, factors, annotated_name, data_interface, pairs, g, pdy,
              meta, datetime_str, output_path, iter=2):
    """Module 6 (short-term): reuse MLLM-CD counterfactual refine, but FREEZE the
    factor set -- iter1 factors are re-used, blocks are NOT re-learned/re-named
    (that omission IS the freeze). New counterfactual samples are re-annotated
    with the iter1 factors; iter2 FCI then refines the graph."""
    uncertain = helper.get_uncertain_factors(annotated_name, g)
    cf_prompt, cf_imgs = helper.generate_causal_refinement_prompt(
        annotated_name, uncertain, data_interface, meta, pairs, pdy=pdy)
    cf_pd = helper.generate_causal_refinement_response(
        cf_prompt, cf_imgs, annotated_name, meta, iter=1, id_start=0)
    cf_pd.index = list(cf_pd["sample_id"])
    # cache cf_pd so a re-run skips the expensive counterfactual (re)generation
    cf_pd.to_csv(os.path.join(output_path, "iter1_cf_pd.csv"), index=False)

    validated = helper.check_counterfactual_generation(
        cf_pd, meta, data_interface, annotated_name, g)

    n_orig = len(meta)
    if not validated.empty:
        meta_r = pd.concat([
            meta, validated[["score", "Review", "Image"]].rename(columns={"Image": "ImagePath"})
        ], ignore_index=True)   # positional index 0..len-1 (needed for annotation)
        print(f"[{dataset}] added {len(validated)} counterfactual samples")
    else:
        print(f"[{dataset}] no valid counterfactuals")
        meta_r = deepcopy(meta).reset_index(drop=True)

    # FREEZE: iter1 factors re-used (no block re-learn/re-name). Quantify the
    # augmented data -- LLM annotation, or MultiView frozen projection of the CFs.
    blocks = load_blocks(dataset)
    ann_r = _quantify_augmented(dataset, helper, meta_r, factors, blocks, output_path, iter, n_orig)

    data_r = ann_r.copy()
    data_r[TARGET] = meta_r["score"].astype(float).values
    names_r = [TARGET] + [f["name"] for f in factors]
    values_r = data_r[names_r].values.astype(float)

    g_r, _ = fci(values_r, alpha=0.05, independence_test_method="kci", verbose=False)
    pdy_r = GraphUtils.to_pydot(g_r, labels=names_r)
    print(pdy_r.to_string())
    _export(g_r, names_r, iter, output_path, datetime_str)
    return factors, names_r, data_r, g_r, pdy_r, meta_r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["MAG", "Lung"], required=True)
    ap.add_argument("--top-n", type=int, default=None,
                    help="override config TOP_N (default: use config)")
    ap.add_argument("--run-seed", type=int, default=C.RUN_SEED)
    ap.add_argument("--data-root", default=C.DATA_ROOT,
                    help="directory containing <dataset>/blocks.pt")
    ap.add_argument("--iter1-only", action="store_true")
    ap.add_argument("--nodes-only", action="store_true",
                    help="node-first: stop after naming, export node-only graph (NP/NR/NF)")
    args = ap.parse_args()
    C.DATA_ROOT = args.data_root
    run_seed = seed_everything(args.run_seed)
    top_n = resolve_top_n(args.top_n)

    # second-resolution timestamp + PID so PARALLEL runs starting the same second
    # never collide on the output folder (env MVEXT_RUN_TAG overrides for clarity).
    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tag = os.environ.get("MVEXT_RUN_TAG") or f"{datetime_str}_p{os.getpid()}"
    ds_dir = "MAG9_mv" if args.dataset == "MAG" else "Lung4_mv"
    # per-run subfolder so repeated/parallel experiments don't overwrite each other.
    output_path = os.path.join("outputs", ds_dir, tag)
    os.makedirs(output_path, exist_ok=True)
    print(f"[{args.dataset}] run outputs -> {output_path}")
    print(f"[{args.dataset}] resolved top_n={top_n} lambda_mode="
          f"{'average' if C.LAMBDA_AVERAGE else 'raw'} run_seed={run_seed}")

    meta = load_meta(args.dataset)
    helper = Helper(datetime_str=datetime_str, N_sample=len(meta), generate=generate,
                    output_path=output_path,
                    dataset=("MAG9" if args.dataset == "MAG" else "Lung4"))

    if args.nodes_only or getattr(C, "NODES_ONLY", False):
        run_nodes_only(args.dataset, helper, datetime_str, output_path,
                       top_n=top_n)
        print(f"[{args.dataset}] nodes-only complete (naming -> NP/NR/NF).")
        return

    factors, annotated_name, data_interface, pairs, g, pdy = run_iter1(
        args.dataset, helper, datetime_str, output_path, top_n=top_n)
    print(f"[{args.dataset}] iter1 complete.")

    if not args.iter1_only:
        refine_mv(args.dataset, helper, factors, annotated_name, data_interface,
                  pairs, g, pdy, meta, datetime_str, output_path, iter=2)
        print(f"[{args.dataset}] iter2 complete.")


if __name__ == "__main__":
    main()
