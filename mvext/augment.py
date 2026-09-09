"""Build the paper's aligned MVCD image and corpus-frequency text views.

Each sample receives three image views and twelve text views. The manifest row
index equals the sample index so downstream folds never desynchronize views.

Usage:
    python -m mvext.augment --dataset MAG
    python -m mvext.augment --dataset Lung
    python -m mvext.augment --dataset both
"""
import argparse
from collections import Counter
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

from mvext import config as C
from mvext.artifacts import write_json

# --- image view ops -------------------------------------------------------


def _to_rgb(img):
    return img.convert("RGB")


def color_view(img):
    """Average-color patch: the whole canvas filled with the mean RGB. Exposes
    color while discarding shape, size and texture."""
    a = np.asarray(_to_rgb(img))
    mean = a.reshape(-1, 3).mean(0).astype(np.uint8)
    h, w, _ = a.shape
    return Image.fromarray(np.tile(mean, (h, w, 1)))


def hpf_view(img, sigma=8):
    """High-pass filter: image minus its Gaussian blur, re-centered. Keeps
    shape/edges/texture (high-freq), discards color/brightness (low-freq)."""
    from PIL import ImageFilter
    a = np.asarray(_to_rgb(img)).astype(np.float32)
    blur = np.asarray(_to_rgb(img).filter(ImageFilter.GaussianBlur(radius=sigma))).astype(np.float32)
    hp = (a - blur) + 128.0
    return Image.fromarray(np.clip(hp, 0, 255).astype(np.uint8))


def _otsu_threshold(gray):
    """Otsu's method on a uint8 grayscale array -> threshold in [0,255]."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    sum_all = np.dot(np.arange(256), hist)
    w_bg = np.cumsum(hist)
    w_fg = total - w_bg
    sum_bg = np.cumsum(np.arange(256) * hist)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_bg = sum_bg / np.maximum(w_bg, 1)
        mean_fg = (sum_all - sum_bg) / np.maximum(w_fg, 1)
        between = w_bg * w_fg * (mean_bg - mean_fg) ** 2
    between[(w_bg == 0) | (w_fg == 0)] = 0
    return int(np.argmax(between))


def silhouette_view(img):
    """Binary silhouette via Otsu thresholding of the grayscale image.

    The foreground (whichever side of the threshold is the minority / darker
    object) is rendered white on black. Generic and dependency-free; a good
    default for both apples-on-background and X-ray fields, tunable later.
    """
    gray = np.asarray(img.convert("L"))
    t = _otsu_threshold(gray)
    fg_dark = gray[gray <= t].size <= gray[gray > t].size  # minority side = object
    mask = (gray <= t) if fg_dark else (gray > t)
    binary = np.where(mask, 255, 0).astype(np.uint8)
    return Image.fromarray(np.stack([binary] * 3, axis=-1))


# --- text views -----------------------------------------------------------


def _mask_keywords(text, keywords):
    """Delete whole-word occurrences of `keywords` (case-insensitive), tidy the
    residual whitespace/punctuation. Grammar is sacrificed, but the view exposes
    the remaining category's vocabulary more purely."""
    if not keywords:
        return text
    pat = re.compile(r"\b(" + "|".join(re.escape(w) for w in keywords) + r")\b",
                     re.IGNORECASE)
    out = pat.sub("", text)
    out = re.sub(r"\s+([,.;:])", r"\1", out)
    out = re.sub(r"\s{2,}", " ", out)
    return out.strip(" ,.;:")


def _base_text(dataset, row):
    if dataset == "MAG":
        return str(row["Review"])
    if dataset == "Lung":
        return f"{str(row['Demographics']).strip()} {str(row['History']).strip()}".strip()
    raise ValueError(dataset)


def rank_keywords(dataset, meta, k=C.TEXT_KEYWORD_K):
    """Rank eligible corpus unigrams by total frequency, deterministically."""
    documents = [_base_text(dataset, row) for _, row in meta.iterrows()]
    analyzer = CountVectorizer(lowercase=True, ngram_range=(1, 1)).build_analyzer()
    tokenized = [analyzer(document) for document in documents]
    term_frequency = Counter(token for document in tokenized for token in document)
    document_frequency = Counter(token for document in tokenized for token in set(document))
    maximum_df = 0.90 * len(documents)
    stop_words = set(ENGLISH_STOP_WORDS)
    excluded = {
        "english_stop_words": sorted(token for token in term_frequency
                                      if token in stop_words),
        "below_min_df": sorted(token for token, count in document_frequency.items()
                               if token not in stop_words and count < 2),
        "above_max_df": sorted(token for token, count in document_frequency.items()
                               if token not in stop_words and count > maximum_df),
    }
    eligible = sorted(token for token in term_frequency
                      if token not in stop_words
                      and 2 <= document_frequency[token] <= maximum_df)
    if len(eligible) < k:
        raise ValueError(f"{dataset}: only {len(eligible)} eligible tokens for K={k}")
    ranking = sorted(eligible, key=lambda token: (-term_frequency[token], token))
    return {
        "schema_version": 1,
        "dataset": dataset,
        "method": "frequency",
        "k": k,
        "settings": {"lowercase": True, "ngram_range": [1, 1],
                     "stop_words": "english", "min_df": 2, "max_df": 0.90,
                     "norm": None},
        "selected": [{"token": token, "score": float(term_frequency[token])}
                     for token in ranking[:k]],
        "ranking": [{"token": token, "score": float(term_frequency[token]),
                     "term_frequency": term_frequency[token],
                     "document_frequency": document_frequency[token]}
                    for token in ranking],
        "excluded": excluded,
    }


def isolate_text_views(text, selected):
    views = {}
    for target in range(len(selected)):
        others = [index for index in range(len(selected)) if index != target]
        for side, indices in (("a", others[::2]), ("b", others[1::2])):
            masked = _mask_keywords(text, {selected[index] for index in indices})
            views[f"txt_iso_{target:02d}_{side}"] = masked
    return views


def _text_views(dataset, row, selected):
    return isolate_text_views(_base_text(dataset, row), selected)


# --- driver ---------------------------------------------------------------


def _save(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


_IMG_OPS = {"img_color": color_view, "img_hpf": hpf_view, "img_silhouette": silhouette_view}


def build_views(dataset, root=C.DATA_ROOT):
    """Generate image view files + text views (view-name driven) and write the
    manifest with one column per view in C.views(dataset) order."""
    out_dir = os.path.join(root, dataset)
    view_dir = os.path.join(out_dir, "views")
    view_names = C.views(dataset)

    if dataset == "MAG":
        meta = pd.read_csv("MAG9.csv")
        src_img = meta["ImagePath"].tolist()
        target = meta["score"].astype(float).tolist()
    elif dataset == "Lung":
        meta = pd.read_csv("Lung.csv", index_col=0)
        src_img = [f"./Lung/{i}.jpg" for i in meta.index]
        target = meta["score"].astype(float).tolist()
    else:
        raise ValueError(dataset)
    meta = meta.reset_index(drop=True)
    specification = rank_keywords(dataset, meta)
    selected = [entry["token"] for entry in specification["selected"]]
    write_json(Path(out_dir) / "text_view_spec.json", specification)

    rows = []
    for i in range(len(src_img)):
        img = Image.open(src_img[i])
        row = {"idx": i, "score": target[i]}
        for name in view_names:
            if name == "img_orig":
                row[name] = src_img[i]
            elif name in _IMG_OPS:
                p = os.path.join(view_dir, name, f"{i}.png")
                _save(_IMG_OPS[name](img), p)
                row[name] = p
        row.update(_text_views(dataset, meta.iloc[i], selected))
        rows.append(row)

    manifest = pd.DataFrame(rows).set_index("idx")[["score"] + view_names]
    os.makedirs(out_dir, exist_ok=True)
    mpath = os.path.join(out_dir, "manifest.csv")
    manifest.to_csv(mpath)
    print(f"[{dataset}] {len(manifest)} samples, {len(view_names)} views -> {mpath}")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["MAG", "Lung", "both"], required=True)
    args = ap.parse_args()
    for ds in (["MAG", "Lung"] if args.dataset == "both" else [args.dataset]):
        build_views(ds)


if __name__ == "__main__":
    main()
