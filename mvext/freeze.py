"""Frozen projection of NEW samples (iter2 counterfactuals) onto the discovered
factors -- the MultiView-quantification path.

A new sample's views are generated the same way as training views, encoded by the
FROZEN CLIP + adapters, projected onto each factor's FROZEN decompose axis, and
ranked against the FIXED original rank map (block_raw). This yields the sample's
rank-0..1 value per factor WITHOUT re-learning or re-ranking the originals -- the
"freeze" (adapters, block axes, and rank map all fixed).
"""
import torch
from PIL import Image

from mvext import config as C
from mvext import augment as A
from mvext.encoders import extract_clip_feature, CLIP_DIM
from mvext.multiview import MultiViewModel, extract_block_projections, average_views


def _sample_features(dataset, image_paths, reviews):
    """Generate the dataset's views for new samples and CLIP-encode them ->
    (n, n_views, CLIP_DIM) tensor matching the cache view order."""
    view_names = C.views(dataset)
    img_views = set(C.image_views(dataset))
    n = len(image_paths)
    feats = torch.zeros(n, len(view_names), CLIP_DIM)
    imgs = [Image.open(p) for p in image_paths]
    for v, name in enumerate(view_names):
        if v in img_views:                                   # image view
            op = A._IMG_OPS.get(name)
            items = [op(im) if op else im for im in imgs]    # PIL images
            f = extract_clip_feature("image", items)
        else:                                                # text pair-mask view
            a, b = name.split("_")[-1]
            items = [A.pairmask_view(dataset, str(r), a, b) for r in reviews]
            f = extract_clip_feature("text", items)
        feats[:, v, :f.shape[1]] = f
    return feats


def _load_frozen_model(blocks):
    cfg = blocks["model_config"]
    model = MultiViewModel(cfg["subsets"], cfg["n_views"], cfg["in_dim"], cfg["enc"],
                           content_sizes=cfg["content_sizes"],
                           variable=cfg.get("variable", False))
    model.load_state_dict(blocks["model_state"])
    model.eval()
    return model


def _rank_against(vals, ref):
    """Rank each of `vals` within the FIXED reference distribution `ref` -> [0,1].
    (# ref <= v) / (len(ref)-1), clamped."""
    ref = ref.reshape(-1)
    denom = max(len(ref) - 1, 1)
    out = torch.empty(len(vals))
    for i, v in enumerate(vals):
        out[i] = (ref <= v).sum().item()
    return (out / denom).clamp(0, 1)


@torch.no_grad()
def project_new_samples(dataset, image_paths, reviews, blocks):
    """Return an (n_new, n_factors) tensor of frozen rank-0..1 factor values,
    columns aligned with blocks['block_subsets']."""
    feats = _sample_features(dataset, image_paths, reviews)
    model = _load_frozen_model(blocks)
    proj = extract_block_projections(model, feats, stats=blocks["feature_stats"])
    cols = []
    for b, subset in enumerate(blocks["block_subsets"]):
        P = average_views(proj[subset])                      # (n, k)
        axis = blocks["block_axes"][b]
        raw = blocks["block_raw"][b]                          # (N, d) fixed map
        cf_raw = (P @ axis.unsqueeze(1)) if axis is not None else P   # (n, d)
        ranked = torch.stack([_rank_against(cf_raw[:, j], raw[:, j])
                              for j in range(cf_raw.shape[1])], dim=1)  # (n, d)
        # collapse with the SAME 1-D reduce axis learned in finalize (PCA-1);
        # fall back to the column mean for blocks saved before this field existed.
        red = blocks.get("block_reduce_axes")
        if red is not None:
            cols.append((ranked * red[b].to(ranked)).sum(1))  # (n,) PCA-1 summary
        else:
            cols.append(ranked.mean(1))                        # 1-D summary per factor
    return torch.stack(cols, dim=1)                          # (n, n_factors)
