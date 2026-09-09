"""Module 2 - per-view encoders.

Two-stage encoder: frozen CLIP feature extraction -> per-view learnable linear
adapter. Because CLIP is frozen, its output for a given view is deterministic,
so we extract it ONCE and cache (N, n_views, D). Training then only runs the
light adapters over the cache.

`extract_clip_feature` is the single swap point for a future extension to
CLIP intermediate features: change what it returns (and rebuild the cache) and
nothing downstream (adapters, MultiView, validity) has to change. The cache is
only valid while the CLIP stage stays frozen; set use_cache=False (feed live
features) if CLIP is ever fine-tuned.

Usage:
    python -m mvext.encoders --dataset MAG      # build the CLIP feature cache
    python -m mvext.encoders --dataset both
"""
import argparse
import os

import clip
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

from mvext import config as C

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_clip_model = None
_clip_preprocess = None


def _get_clip():
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=_DEVICE)
        _clip_model.eval()
    return _clip_model, _clip_preprocess


# --- REPLACEABLE feature-extraction point --------------------------------
# The final CLIP projection maps into the shared contrastive space (holistic);
# pre-projection / token-mean features (config.IMAGE_FEATURE / TEXT_FEATURE) keep
# more factor-specific detail. Dims differ per modality (image up to 768, text
# 512); build_clip_cache zero-pads to a common width. Keep this frozen while the
# cache is used.
CLIP_DIM = 768   # max width (image pre-projection); text is padded up to this


def _image_feature(model, images, kind):
    """kind: 'final'|'preproj'|'tokens_mean'. Replicates VisionTransformer.forward
    up to the chosen point (before the 768->512 projection for non-final)."""
    if kind == "final":
        return model.encode_image(images).float()
    v = model.visual
    x = v.conv1(images.type(v.conv1.weight.dtype))            # (B, 768, gh, gw)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # (B, P, 768)
    cls = v.class_embedding.to(x.dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
    x = torch.cat([cls, x], dim=1) + v.positional_embedding.to(x.dtype)
    x = v.ln_pre(x).permute(1, 0, 2)
    x = v.transformer(x).permute(1, 0, 2)                     # (B, 1+P, 768)
    if kind == "preproj":
        return v.ln_post(x[:, 0, :]).float()                 # CLS, pre-projection
    if kind == "tokens_mean":
        return v.ln_post(x[:, 1:, :]).mean(1).float()        # patch tokens mean
    raise ValueError(kind)


def _text_feature(model, tokens, kind):
    """kind: 'final'|'preproj'|'tokens_mean'. Replicates encode_text up to the
    chosen point (before the 512->512 text_projection for non-final)."""
    if kind == "final":
        return model.encode_text(tokens).float()
    x = model.token_embedding(tokens).type(model.dtype)
    x = x + model.positional_embedding.type(model.dtype)
    x = x.permute(1, 0, 2)
    x = model.transformer(x).permute(1, 0, 2)
    x = model.ln_final(x).type(model.dtype)                   # (B, 77, 512)
    if kind == "preproj":
        eot = tokens.argmax(dim=-1)                           # EOT position
        return x[torch.arange(x.shape[0]), eot].float()
    if kind == "tokens_mean":
        mask = (tokens != 0).unsqueeze(-1).type(x.dtype)      # non-pad tokens
        return ((x * mask).sum(1) / mask.sum(1).clamp(min=1)).float()
    raise ValueError(kind)


@torch.no_grad()
def extract_clip_feature(kind, items, batch_size=64, feature_kind=None):
    """kind in {'image','text'}; items = list of image paths or strings.
    Returns a (len(items), D) float tensor on CPU (D=768 image / 512 text for
    the pre-projection variants). feature_kind overrides config default."""
    model, preprocess = _get_clip()
    fk = feature_kind or (C.IMAGE_FEATURE if kind == "image" else C.TEXT_FEATURE)
    out = []
    for s in range(0, len(items), batch_size):
        chunk = items[s:s + batch_size]
        if kind == "image":
            # items may be file paths OR already-loaded PIL images (frozen projection)
            pil = [(im if isinstance(im, Image.Image) else Image.open(im)).convert("RGB")
                   for im in chunk]
            batch = torch.stack([preprocess(im) for im in pil]).to(_DEVICE)
            feat = _image_feature(model, batch, fk)
        elif kind == "text":
            tokens = clip.tokenize([str(t) for t in chunk], truncate=True).to(_DEVICE)
            feat = _text_feature(model, tokens, fk)
        else:
            raise ValueError(kind)
        out.append(feat.float().cpu())
    return torch.cat(out, 0)


# --- cache ---------------------------------------------------------------


def _cache_path(dataset, root=C.DATA_ROOT):
    return os.path.join(root, dataset, "clip_cache.pt")


def build_clip_cache(dataset, root=C.DATA_ROOT):
    """Extract CLIP features for every configured view and cache them."""
    manifest = pd.read_csv(os.path.join(root, dataset, "manifest.csv"), index_col=0)
    n = len(manifest)
    view_names = C.views(dataset)
    img_views = set(C.image_views(dataset))
    feats = torch.zeros(n, len(view_names), CLIP_DIM)   # zero-pad to common width
    for v, name in enumerate(view_names):
        kind = "image" if v in img_views else "text"
        f = extract_clip_feature(kind, manifest[name].tolist())   # (n, D_view)
        feats[:, v, :f.shape[1]] = f
        print(f"[{dataset}] view {v} ({name}, {kind}) encoded  dim={f.shape[1]}")
    path = _cache_path(dataset, root)
    torch.save({
        "features": feats,               # (N, n_views, CLIP_DIM)
        "view_names": view_names,
        "dim": CLIP_DIM,
        "idx": list(manifest.index),
        "score": torch.tensor(manifest["score"].values, dtype=torch.float),
    }, path)
    print(f"[{dataset}] cache {tuple(feats.shape)} -> {path}")
    return path


def load_cache(dataset, root=C.DATA_ROOT):
    cache = torch.load(_cache_path(dataset, root), map_location="cpu", weights_only=False)
    expected = C.views(dataset)
    if cache.get("view_names") != expected:
        raise ValueError(f"stale {dataset} CLIP cache; rerun the prepare stage")
    return cache


# --- adapters ------------------------------------------------------------


class ViewEncoder(nn.Module):
    """One independent linear adapter per view: CLIP feature (D) -> encoding.

    CLIP is not part of this module (features arrive pre-extracted, cached or
    live), so only the adapters are trainable.
    """

    def __init__(self, n_views, in_dim=CLIP_DIM, out_dim=C.ENCODING_SIZE):
        super().__init__()
        self.adapters = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(n_views)])

    def forward(self, feats):
        """feats: (n_views, batch, in_dim) -> (n_views, batch, out_dim)."""
        return torch.stack([self.adapters[k](feats[k]) for k in range(len(self.adapters))], dim=0)

    def encode_view(self, view_idx, feat):
        """Adapter for a single view; feat: (batch, in_dim) -> (batch, out_dim)."""
        return self.adapters[view_idx](feat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["MAG", "Lung", "both"], required=True)
    args = ap.parse_args()
    for ds in (["MAG", "Lung"] if args.dataset == "both" else [args.dataset]):
        build_clip_cache(ds)


if __name__ == "__main__":
    main()
