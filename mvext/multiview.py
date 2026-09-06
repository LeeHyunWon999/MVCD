"""Module 3 - MultiView block discovery.

Ports the essential MultiView-CRL machinery to our setting: per-view adapters
(from mvext.encoders) + a per-subset learnable content selector, trained with a
multi-view InfoNCE that aligns each subset's shared 1-D content block across its
views while contrasting against other samples.

Deviations from multiview-crl (see IMPLEMENTATION_LOG.md), all deliberate:
  - Selector logits are a FREE learnable parameter per subset `(n_subsets, enc)`
    (design A), not the shared batch-mean `hz.mean(0)` of the original. Our 8
    subsets are arbitrary (non-nested), so each needs its own selection.
  - Selection is applied as a differentiable soft-mask weighted sum (so gradients
    reach the free logits), not integer indexing (`torch.where`, non-diff).
  - Similarity is -L2^2 (InfoNCE a la the numerical LpSimCLR), not cosine: for
    1-D content the cosine of scalars degenerates to a sign.

Output contract (consumed by modules 4/5): extract_block_projections returns
per-view projections c[subset][view] -> (N, content_size), WITHOUT averaging
views (permutation test needs per-view; downstream/seed-stability average).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mvext import config as C
from mvext.encoders import ViewEncoder, CLIP_DIM

_EPS = 1e-6


def topk_gumbel_softmax(logits, k, tau=1.0, hard=True):
    """Top-k Gumbel-Softmax over the last dim, per row. logits: (R, D).
    Returns a (R, D) soft mask (~k-hot); straight-through hard if hard=True.
    Ported from multiview-crl utils.topk_gumbel_softmax (row-batched)."""
    g = torch.distributions.gumbel.Gumbel(
        torch.zeros_like(logits), torch.ones_like(logits)).sample()
    l = logits + g
    khot = torch.zeros_like(l)
    onehot_approx = torch.zeros_like(l)
    for _ in range(k):
        khot_mask = torch.clamp(1.0 - onehot_approx, min=_EPS)
        l = l + torch.log(khot_mask)
        onehot_approx = F.softmax(l / tau, dim=-1)
        khot = khot + onehot_approx
    if hard:
        top = torch.topk(khot, k, dim=-1).indices
        hard_mask = torch.zeros_like(khot).scatter_(-1, top, 1.0)
        khot = (hard_mask - khot).detach() + khot  # straight-through
    return khot


class MultiViewModel(nn.Module):
    """Per-view adapters + per-subset content selectors."""

    def __init__(self, subsets, n_views, in_dim=CLIP_DIM,
                 enc=C.ENCODING_SIZE, content_sizes=None, variable=False):
        super().__init__()
        self.subsets = [tuple(s) for s in subsets]
        # PER-SUBSET content size (like the MultiView authors' content_size_dict):
        # each subset selects its own k dims. content_sizes is a list aligned with
        # subsets; falls back to a uniform default if not given.
        if content_sizes is None:
            content_sizes = [C.CONTENT_SIZE_UNIFORM] * len(self.subsets)
        self.content_sizes = [int(k) for k in content_sizes]
        self.variable = variable                 # learn k via L0 gates (Thm 3.8)
        self.encoder = ViewEncoder(n_views, in_dim, enc)
        # design A: independent learnable selection logits per subset. In variable
        # mode these are per-dim GATE logits (init ~0 -> start ~half active, let the
        # alignment loss keep shared dims and the L0 penalty trim the rest).
        self.sel_logits = nn.Parameter(torch.randn(len(self.subsets), enc) * 0.01)

    def encode(self, feats):
        """feats: (n_views, batch, in_dim) -> hz (n_views, batch, enc)."""
        return self.encoder(feats)

    def soft_masks(self, tau=1.0):
        """Per-subset differentiable content mask, (n_subsets, enc).
        topk mode: subset b is ~content_sizes[b]-hot. variable mode: independent
        straight-through per-dim gates (sigmoid>0.5), so k is learned per subset."""
        if self.variable:
            g = torch.sigmoid(self.sel_logits)
            hard = (g > 0.5).float()
            return hard + (g - g.detach())       # straight-through
        masks = [topk_gumbel_softmax(self.sel_logits[b:b + 1], k, tau, hard=True)[0]
                 for b, k in enumerate(self.content_sizes)]
        return torch.stack(masks, 0)

    def l0_penalty(self):
        """Expected number of active gates (variable mode) -- sparsity term."""
        return torch.sigmoid(self.sel_logits).sum() / len(self.subsets) if self.variable \
            else self.sel_logits.new_zeros(())

    def hard_indices(self):
        """Deterministic selected dims per subset: list of index lists.
        variable mode -> dims with gate>0.5 (>=1 kept); topk -> top content_sizes[b]."""
        if self.variable:
            out = []
            for b in range(len(self.subsets)):
                idx = (torch.sigmoid(self.sel_logits[b]) > 0.5).nonzero().flatten().tolist()
                if not idx:                       # never empty: keep the top gate
                    idx = [int(self.sel_logits[b].argmax())]
                out.append(sorted(idx))
            return out
        return [sorted(torch.topk(self.sel_logits[b], k).indices.tolist())
                for b, k in enumerate(self.content_sizes)]

    def project_soft(self, hz, masks):
        """Differentiable per-subset per-view content projection, aligned by
        InfoNCE across views. Returns list over subsets of list over views of
        (batch, enc) MASKED vectors: the (~content_size) selected dims carry the
        value, the rest are ~0. InfoNCE(-L2^2) on these aligns EACH selected dim
        across views (a true multi-dim shared block).

        (Previously this summed the masked dims to a scalar -> the block collapsed
        to 1-D regardless of content_size, so content_size>1 wasn't really learned.
        For content_size==1 the two are mathematically identical.)"""
        out = []
        for b, subset in enumerate(self.subsets):
            w = masks[b]                       # (enc,), ~content_size-hot
            out.append([hz[k] * w for k in subset])   # each (batch, enc)
        return out

    def project_hard(self, hz):
        """Inference projection using hard selected dims. Same structure."""
        idx = self.hard_indices()
        out = []
        for b, subset in enumerate(self.subsets):
            dims = idx[b]
            out.append([hz[k][:, dims] for k in subset])
        return out


def infonce_l2(view_projs, tau=1.0, dim_norm=1, sim_mode="l2"):
    """Multi-view InfoNCE over one subset's views.
    view_projs: list (len=|subset|) of (batch, D) masked vectors (k active dims).
    sim_mode: "l2" -> -||ca-cb||^2 similarity (default); "dot" -> <ca,cb> inner
    product. dim_norm divides the similarity so its scale is invariant to the
    block dimensionality k (pass k to balance mixed per-subset sizes).
    """
    n = len(view_projs)
    d = view_projs[0].shape[0]
    total = view_projs[0].new_zeros(())
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            ca = view_projs[a].unsqueeze(1)    # (d,1,dim)
            cb = view_projs[b].unsqueeze(0)    # (1,d,dim)
            if sim_mode == "dot":
                sim = (ca * cb).sum(-1) / (tau * dim_norm)       # (d,d) inner product
            else:
                sim = -((ca - cb) ** 2).sum(-1) / (tau * dim_norm)  # (d,d) -L2^2
            total = total + F.cross_entropy(sim, torch.arange(d, device=sim.device))
    return total / (n * (n - 1))


def barlow_l2(view_projs, mask, lam=1.0, dim_norm=1, eps=1e-5):
    """BarlowTwins over one subset's views: push the cross-correlation matrix of
    the (batch-normalized) content projections toward identity -- on-diagonal =
    alignment, off-diagonal = redundancy reduction (entropy proxy). Only the
    subset's ACTIVE dims (mask>0) contribute; loss is normalized by dim_norm.
    view_projs: list of (batch, D) masked vectors; mask: (D,) ~k-hot selector."""
    n = len(view_projs)
    D = view_projs[0].shape[-1]
    m = (mask > 0).float()                                  # (D,) active dims
    mm = m[:, None] * m[None, :]                            # (D,D) active pairs
    eye = torch.eye(D, device=view_projs[0].device)

    def bn(z):                                              # batch-normalize per dim
        return (z - z.mean(0, keepdim=True)) / (z.std(0, keepdim=True) + eps)

    total = view_projs[0].new_zeros(())
    for a in range(n):
        for b in range(a + 1, n):
            za, zb = bn(view_projs[a]), bn(view_projs[b])
            c = (za.T @ zb) / za.shape[0]                   # (D,D) cross-correlation
            diff = (c - eye) ** 2
            on = (diff * eye * mm).sum()                    # (c_dd - 1)^2 over active
            off = (diff * (1 - eye) * mm).sum()             # c_de^2 over active pairs
            total = total + (on + lam * off)
    return total / (n * (n - 1) / 2) / max(dim_norm, 1)


def standardize_features(feats, stats=None):
    """Optionally z-score features over samples per (view, dim). OFF by default
    (see IMPLEMENTATION_LOG.md: magnitude info is intentionally preserved)."""
    if stats is None:
        mean = feats.mean(0, keepdim=True)
        std = feats.std(0, keepdim=True).clamp_min(_EPS)
        stats = (mean, std)
    mean, std = stats
    return (feats - mean) / std, stats


def train_multiview(features, subsets, enc=C.ENCODING_SIZE,
                    content_sizes=None, variable=False, steps=2000, lr=1e-3, tau=1.0,
                    batch_size=None, standardize=False, seed=0, device=None,
                    log_every=250, verbose=True):
    """Train adapters + selectors on a feature cache.

    features: (N, n_views, D) tensor (raw CLIP features from mvext.encoders).
    content_sizes: per-subset content dims (list aligned with subsets); None ->
    uniform default. Returns (model, info) with loss curve, selected dims, stats.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    feats = features.to(device).float()
    stats = None
    if standardize:
        feats, stats = standardize_features(feats)

    N, n_views, D = feats.shape
    model = MultiViewModel(subsets, n_views, D, enc, content_sizes, variable=variable).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    bs = batch_size or N
    losses = []
    model.train()
    for step in range(steps):
        idx = torch.randperm(N, device=device)[:bs] if bs < N else torch.arange(N, device=device)
        batch = feats[idx].permute(1, 0, 2)      # (n_views, bs, D)
        hz = model.encode(batch)                 # (n_views, bs, enc)
        masks = model.soft_masks(tau)
        projs = model.project_soft(hz, masks)
        # per-subset loss; optionally normalize by content size (mixed-k balance)
        terms = []
        for b, p in enumerate(projs):
            dnorm = model.content_sizes[b] if C.LOSS_DIM_NORMALIZE else 1
            if C.LOSS == "barlow":
                terms.append(barlow_l2(p, masks[b], C.BARLOW_LAMBDA, dnorm))
            else:
                terms.append(infonce_l2(p, tau, dnorm, sim_mode=getattr(C, "LOSS_SIM", "l2")))
        loss = sum(terms) / len(terms)
        if model.variable:                       # L0 sparsity: learn k per subset
            loss = loss + C.VARIABLE_L0_ALPHA * model.l0_penalty()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        losses.append(loss.item())
        if verbose and (step % log_every == 0 or step == steps - 1):
            print(f"  step {step:5d}  loss {loss.item():.4f}")

    model.eval()
    info = {
        "losses": losses,
        "selected_dims": model.hard_indices(),
        "feature_stats": stats,
        "seed": seed,
        "standardize": standardize,
    }
    return model, info


@torch.no_grad()
def extract_block_projections(model, features, stats=None, device=None):
    """Per-view projections for every subset, using HARD selection.

    Returns dict: subset_tuple -> {view_idx -> (N, content_size) cpu tensor}.
    Views are NOT averaged here (permutation test needs per-view).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    feats = features.to(device).float()
    if stats is not None:
        feats, _ = standardize_features(feats, stats)
    hz = model.encode(feats.permute(1, 0, 2))    # (n_views, N, enc)
    projs = model.project_hard(hz)
    out = {}
    for b, subset in enumerate(model.subsets):
        out[subset] = {k: projs[b][j].cpu() for j, k in enumerate(subset)}
    return out


def average_views(subset_proj):
    """Collapse {view -> (N,d)} to a per-sample (N,d) block value by averaging
    participating views. Used by downstream + seed-stability (NOT permutation)."""
    vs = list(subset_proj.values())
    return torch.stack(vs, 0).mean(0)
