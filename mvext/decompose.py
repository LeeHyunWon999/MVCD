"""Sparse local difference decomposition (follow-up feature).

Splits a high-dim (entangled) block into 1-D factor axes. Rationale (plan
§3-4): between nearby samples only a few factors change, so the DIRECTIONS of
local difference vectors cluster around the block's underlying atomic axes.
Clustering those directions recovers axes we can project onto, giving 1-D
factors that then reuse the existing rank-norm -> dedup -> naming pipeline.

Not guaranteed to be truly atomic; treated as "factors useful for reasoning"
(the causal-discovery step judges them downstream).
"""
import numpy as np

from mvext import config as C


def _unit_sign_fixed(v):
    """Unit vectors with sign canonicalized (axis direction is arbitrary): flip
    each so its largest-magnitude component is positive."""
    n = np.linalg.norm(v, axis=1, keepdims=True)
    u = v / np.clip(n, 1e-9, None)
    lead = np.argmax(np.abs(u), axis=1)
    signs = np.sign(u[np.arange(len(u)), lead])
    signs[signs == 0] = 1.0
    return u * signs[:, None]


def _local_diff_directions(P, k=C.DECOMP_KNN):
    """Unit, sign-fixed difference directions between each sample and its k
    nearest neighbors in block-projection space P (N x D)."""
    from sklearn.neighbors import NearestNeighbors
    N = len(P)
    k = min(k, N - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(P)
    _, idx = nn.kneighbors(P)
    diffs = []
    for i in range(N):
        for j in idx[i, 1:]:
            diffs.append(P[i] - P[j])
    return _unit_sign_fixed(np.asarray(diffs))


def decompose_block(P, max_axes=None, k=C.DECOMP_KNN,
                    min_support=C.DECOMP_MIN_SUPPORT, tightness=C.DECOMP_TIGHTNESS):
    """Decompose a block projection P (N x D) into 1-D factor axes.

    Returns (values, axes): values is (N x n_axes) projections onto the kept
    axes; axes is (n_axes x D). If P is effectively 1-D or no split is
    supported, returns the single dominant (PCA) axis.
    """
    from sklearn.cluster import KMeans
    P = np.asarray(P, dtype=float)
    N, D = P.shape
    if D == 1:
        return P.copy(), np.ones((1, 1))

    dirs = _local_diff_directions(P, k)
    max_axes = max_axes or (C.DECOMP_MAX_AXES or D)

    best = None
    for n_clusters in range(1, min(max_axes, D) + 1):
        if n_clusters == 1:
            axes = _dominant_axis(P)[None]
            support = np.array([1.0])
            tight = np.array([np.abs(dirs @ axes[0]).mean()])
        else:
            km = KMeans(n_clusters=n_clusters, n_init=5, random_state=0).fit(dirs)
            axes = _unit_sign_fixed(km.cluster_centers_)
            labels = km.labels_
            support, tight = [], []
            for c in range(n_clusters):
                m = labels == c
                support.append(m.mean())
                tight.append(np.abs(dirs[m] @ axes[c]).mean() if m.any() else 0.0)
            support, tight = np.array(support), np.array(tight)
        keep = (support >= min_support) & (tight >= tightness)
        kept_axes = axes[keep]
        # prefer more axes as long as all kept clusters are well-supported/tight
        if len(kept_axes) >= 1:
            best = (kept_axes, len(kept_axes))
        else:
            break
    if best is None:
        kept_axes = _dominant_axis(P)[None]
    else:
        kept_axes = best[0]
    values = P @ kept_axes.T                       # (N, n_axes)
    return values, kept_axes


def _dominant_axis(P):
    Pc = P - P.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(Pc, full_matrices=False)
    a = vt[0]
    if np.abs(a)[np.argmax(np.abs(a))] and a[np.argmax(np.abs(a))] < 0:
        a = -a
    return a
