"""Module 5 - block-based CFD replacement (A1: new module, originals untouched).

Replaces MLLM-CD's CLIP-distance contrastive-pair discovery with MultiView
block-based discovery, but keeps MLLM-CD's factor format and downstream stages
(annotation, FCI). For each validated+deduped block it selects sample pairs that
differ most in THAT block while differing least in the others, shows the pairs'
ORIGINAL image+text to the MLLM, and asks for the dominant differentiating
factor -- yielding a {name, criteria} factor per block. The result plugs into
Helper.annotate_all_samples (LLM 1/0/-1) exactly like the original CFD.

The annotation step (LLM 1/0/-1) is kept separate so it can later be swapped for
block-projection quantification (follow-up goal); see quantify_* below.
"""
import torch

from mvext import config as C
from mvext.encoders import load_cache  # noqa: F401  (kept for parity/imports)

def load_blocks(dataset, root=None):
    root = root or C.DATA_ROOT
    return torch.load(f"{root}/{dataset}/blocks.pt", map_location="cpu", weights_only=False)


def _pair_diff(v, mode):
    """(N,N) pairwise DIFFERENCE matrix (large = dissimilar) under `mode`:
    'l2dist' -> L2 distance (default naming); 'l2sq' -> squared L2 (== -loss sim
    when LOSS_SIM='l2', monotone with L2); 'dot' -> negative inner product
    (== -loss sim when LOSS_SIM='dot')."""
    if mode == "dot":
        return -(v @ v.T)
    d = torch.cdist(v, v)
    return d * d if mode == "l2sq" else d


def _naming_diff_mode():
    """Which pairwise metric naming uses (B6): match the loss's sim if
    NAMING_USE_LOSS_SIM, else the fixed L2 distance."""
    if getattr(C, "NAMING_USE_LOSS_SIM", False):
        return "l2sq" if getattr(C, "LOSS_SIM", "l2") == "l2" else "dot"
    return "l2dist"


def resolve_top_n(cli_top_n=None):
    return C.TOP_N if cli_top_n is None else cli_top_n


def _effective_lambda(n_other):
    if n_other == 0:
        return 0.0
    return 1.0 / n_other if C.LAMBDA_AVERAGE else 1.0


def select_block_pairs(block_values, block_idx, top_n=None):
    """Pairs (i,j) with large difference in block_idx and small difference in the
    rest. `block_values` is a list of (N, d_i) tensors per factor (raw P^{(b)} or
    rank-normalized, per RANK_NORMALIZE). The pairwise metric follows the loss's
    similarity when NAMING_USE_LOSS_SIM (B6), else L2 distance. The other-block
    score always uses d_target - lambda*sum(d_other). LAMBDA_AVERAGE selects
    lambda=1/(M-1); otherwise lambda=1.
    Returns [(i,j,score),...] desc."""
    top_n = resolve_top_n(top_n)
    mode = _naming_diff_mode()
    diff_tgt = _pair_diff(block_values[block_idx], mode)                       # (N,N)
    diff_oth = torch.zeros_like(diff_tgt)
    n_oth = 0
    for b, vb in enumerate(block_values):
        if b != block_idx:
            diff_oth = diff_oth + _pair_diff(vb, mode)
            n_oth += 1
    score = diff_tgt - _effective_lambda(n_oth) * diff_oth
    N = diff_tgt.shape[0]
    iu = torch.triu_indices(N, N, offset=1)
    vals = score[iu[0], iu[1]]
    order = torch.argsort(vals, descending=True)[:top_n]
    return [(int(iu[0][k]), int(iu[1][k]), float(vals[k])) for k in order]


def _block_modality(subset, dataset):
    """'image', 'text', or 'both' for a block's view subset."""
    iv, tv = set(C.image_views(dataset)), set(C.text_views(dataset))
    img = all(v in iv for v in subset)
    txt = all(v in tv for v in subset)
    return "image" if img else "text" if txt else "both"


def build_naming_prompt(helper, meta, pairs, dataset, subset, used_factors=None):
    """Prompt + image path list for one block's pairs, asking for the single
    dominant differentiating factor in MLLM-CD's template. Only the block's own
    modality is shown (image block -> images only; text block -> reviews only)
    so the model isn't misled by the modality that doesn't carry this factor.
    `used_factors` (already-named factors) is shown so the model names THIS block
    distinctly -- these blocks are mathematically separated, so avoid collapsing
    them under one umbrella name unless they truly capture the same aspect."""
    role = ("You are an expert food analyst specializing in apple evaluation."
            if dataset == "MAG" else
            "You are an expert medical analyst specializing in lung disease evaluation."
            if dataset == "Lung" else
            "You are an expert analyst specializing in the given samples.")
    modality = _block_modality(subset, dataset)
    show_img = modality in ("image", "both")
    show_txt = modality in ("text", "both")

    parts = ["# Block Contrastive Analysis\n",
             "I'm showing you pairs of samples that differ predominantly along a "
             "single hidden factor (they are similar in all other factors).\n"]
    image_paths = []
    for idx, (i, j, score) in enumerate(pairs):
        parts.append(f"\n## Pair {idx + 1}\n")
        for tag, s in [("A", i), ("B", j)]:
            parts.append(f"### Sample {tag} (ID: {s})\n")
            parts.append(f"- **Score**: {meta.iloc[s]['score']}\n")
            if show_txt:
                parts.append(f"- **Review**: {meta.iloc[s]['Review']}\n")
            if show_img:
                image_paths.append(meta.iloc[s]['ImagePath'])
                parts.append(f"- **Image**: [Image {len(image_paths)}] (provided with this prompt)\n")

    modality_hint = ("the apple images" if modality == "image" and dataset == "MAG"
                     else "the images" if modality == "image"
                     else "the review texts" if modality == "text"
                     else "the images and texts")
    distinct_hint = ""
    if used_factors:
        distinct_hint = ("\n4. These already-named, mathematically-separated factors exist -- "
                         "name THIS one DISTINCTLY from them (do not reuse or collapse into "
                         "them unless it is truly the identical aspect):\n\t- "
                         + "\n\t- ".join(used_factors) + "\n")
    parts.append(f"""
# Task: Name the Dominant Factor

{role} Across these pairs, one concrete factor visible in {modality_hint} changes
while the others stay roughly constant. Identify that single dominant factor.

1. It must be one concrete, observable aspect (no umbrella terms like "overall quality").
2. Give it a concise, specific 2-4 word NAME describing that aspect
   (e.g. "Surface Decay", "Sweetness Level"); do not write the literal word "Factor".
3. Give clear positive / neutral / negative criteria.{distinct_hint}

# Output Format
**Part 1**: Briefly explain what consistently differs across the pairs.
**Part 2**: Output exactly one factor using this exact template, with the name on
the first line in double asterisks:
'''
**<concise factor name>**
- 1: [Positive Criterion]
- 0: [Otherwise; or not mentioned]
- -1: [Negative Criterion]
'''
""")
    return "".join(parts), image_paths


def _merge_distinct_first(helper, factors, iter=1):
    """LLM dedup biased toward KEEPING factors distinct: merge only when clearly
    the same aspect. Softer than the default 'merge similar' so value-distinct
    blocks aren't collapsed under one umbrella name."""
    ftext = ""
    for i, f in enumerate(factors):
        c = f["criteria"]
        ftext += (f"Factor {i+1}: {f['name']}\n- 1: {c.get('positive','N/A')}\n"
                  f"- 0: {c.get('neutral','N/A')}\n- -1: {c.get('negative','N/A')}\n\n")
    prompt = f"""# Factor Refinement (keep factors distinct)

These factors were each derived from a SEPARATE, mathematically-isolated signal:

{ftext}
Please:
1. FIRST keep every factor DISTINCT and specific to one concrete aspect; prefer to
   RETAIN separate factors. Rename for clarity so each is clearly different.
2. Merge two factors ONLY if they clearly describe the SAME underlying aspect
   (not merely related). Do not create umbrella factors like "overall quality".
3. Output the resulting list (as many as remain distinct).

Output each factor in this exact format:
'''
**Factor Name**
- 1: [Positive Criterion]
- 0: [Otherwise; or not mentioned]
- -1: [Negative Criterion]
'''
"""
    resp = helper.generate(prompt)
    return helper.extract_factors_from_response(resp["text"])


def discover_factors_from_blocks(helper, meta, blocks, dataset, top_n=None,
                                 iter=1, quant=None, distinct_first=None):
    """Name each deduped block via block-contrastive naming.

    quant="llm": name blocks, then LLM-merge (distinct-first if enabled) -> factors
      whose per-sample values come from LLM annotation downstream.
    quant="multiview": NO merge -- each deduped block IS a factor (its value is the
      block representation); the LLM only names them, sequentially and distinctly.
    Returns (factors, pairs_by_block). For multiview, factors align 1:1 with
    block_subsets order (so their block_rank01 columns can be used as values)."""
    quant = quant or C.QUANT
    distinct_first = C.NAMING_DISTINCT_FIRST if distinct_first is None else distinct_first
    block_values = blocks["block_values"]
    block_subsets = blocks["block_subsets"]
    per_block, all_factors, pairs_by_block = [], [], {}
    used = []                                    # already-named (for distinct hint)
    for b, subset in enumerate(block_subsets):
        pairs = select_block_pairs(block_values, b, top_n)
        pairs_by_block[subset] = pairs
        hint = used if distinct_first else None
        prompt, imgs = build_naming_prompt(helper, meta, pairs, dataset, subset, used_factors=hint)
        resp = helper.generate(prompt, img_path=imgs)
        facs = helper.extract_factors_from_response(resp["text"])
        fac = facs[0] if facs else {"name": f"Factor {b+1}", "criteria": {}}
        print(f"  block {subset}: {fac['name']}")
        per_block.append(fac)
        all_factors.extend(facs)
        used.append(fac["name"])

    if quant == "multiview":
        return per_block, pairs_by_block          # one factor per block, no merge
    if distinct_first:
        return _merge_distinct_first(helper, all_factors, iter=iter), pairs_by_block
    return helper.deduplicate_factors_with_llm(all_factors, iter=iter), pairs_by_block


# --- annotation seam (swappable; see follow-up: replace with block projection) --


def annotate_llm(helper, meta, factors, iter=1):
    """Short-term quantification: MLLM-CD's LLM 1/0/-1 annotation."""
    import os
    import pandas as pd
    N = len(meta)
    names = [f["name"] for f in factors]
    csv = os.path.join(helper.output_path, f"iter{iter}_annotation_results.csv")
    if os.path.exists(csv):
        cached = pd.read_csv(csv, index_col=0)
        if list(cached.columns) == names and len(cached) == N:
            return cached
        print("  (stale annotation cache: factor set changed -> re-annotating)")
    annotations = helper.annotate_all_samples(meta, factors)
    df = pd.DataFrame(0, index=range(N), columns=[f["name"] for f in factors])
    for sid, fvals in annotations.items():
        for fidx, val in fvals.items():
            df.loc[sid, factors[fidx]["name"]] = val
    return df


def annotate_block_projection(blocks, factors):
    """Follow-up quantification stub: use block_rank01 directly as the FCI values
    (skip the LLM). Wired later; kept here to mark the seam."""
    raise NotImplementedError("follow-up: quantify via block projection, not LLM")
