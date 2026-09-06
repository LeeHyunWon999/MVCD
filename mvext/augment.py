"""Module 1 - view augmentation.

Turns each sample's (image, text) into 7 aligned views (4 image + 3 text) and
writes a manifest that is the single source of truth for sample<->view
alignment. The manifest row index equals the sample index; downstream folds use
this index with shuffle=False so a sample never desyncs from its views.

Usage:
    python -m mvext.augment --dataset MAG
    python -m mvext.augment --dataset Lung
    python -m mvext.augment --dataset both
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
from PIL import Image

from mvext import config as C

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


# --- text view ops --------------------------------------------------------


def _split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", str(text).strip())
    return [p for p in parts if p.strip()]


# MAG text is split into 5 semantic categories (mapping to GT text factors:
# taste, aroma, juiciness, nutrition, recmd/market). Each category VIEW keeps its
# own vocabulary and removes the OTHER categories' keywords (word-level masking),
# so the encoder sees that category's signal more purely -- even in mixed
# sentences. Keyword sets are corpus-frequency derived.
MAG_KW = {
    "taste": {
        "sweet", "sweetness", "sour", "tart", "taste", "flavor", "bitter",
        "bland", "tangy", "palate", "sugary", "acidic", "flavour",
    },
    "aroma": {
        "aroma", "fragrant", "musty", "rotten", "odor", "odour", "scent",
        "smell", "notes", "pungent", "fermented", "aromatic", "fragrance",
    },
    "juiciness": {
        "moisture", "moist", "juicy", "juiciness", "dry", "texture", "crisp",
        "mealy", "watery", "refreshing", "firm", "tender", "crunchy", "succulent",
    },
    "nutrition": {
        "nutritional", "nutrient", "nutrients", "nutrition", "fiber", "vitamin",
        "dietary", "essential", "protein", "mineral", "antioxidant", "healthy",
        "wholesome",
    },
    "market": {
        "market", "potential", "financial", "losses", "consumer", "expectations",
        "sales", "returns", "appeal", "risk", "unacceptable", "commercial",
        "batch", "profit", "value", "demand", "price", "revenue", "clientele",
        "supply", "standards", "commendable", "sell", "buyer",
    },
}
MAG_TEXT_VIEWS = ["taste", "aroma", "juiciness", "nutrition", "market"]

# Lung text split into the 4 GROUND-TRUTH text factors (gender/age/smoking/lesion;
# diagnosis is the target). gender & age appear in every Demographics line, so
# coverage is far better than the earlier frequency-based categories. Codes
# g/a/s/l in the config view names txt_pm_XY.
LUNG_KW = {
    "gender": {
        "male", "female", "man", "woman", "men", "women", "gentleman", "lady",
        "boy", "girl", "gender", "sex",
    },
    "age": {
        "y", "yo", "year", "years", "old", "aged", "age", "ages", "elderly",
        "young", "adult", "adults", "decade", "newborn", "infant",
    },
    "smoking": {
        "smoker", "smokers", "smoking", "smoke", "smoked", "tobacco", "cigarette",
        "cigarettes", "pack", "packs", "nicotine", "nonsmoker",
    },
    "lesion": {
        "lesion", "lesions", "mass", "masses", "nodule", "nodules", "tumor",
        "tumour", "cancer", "carcinoma", "malignant", "malignancy", "metastasis",
        "metastatic", "neoplasm", "opacity", "opacities", "consolidation",
        "infiltrate", "effusion", "abnormal", "abnormality", "radiograph",
        "biopsy", "fibrosis", "lymphoma",
    },
}

# per-dataset keyword categories + single-letter code -> category name.
_KW = {"MAG": MAG_KW, "Lung": LUNG_KW}
_CAT_CODE = {
    "MAG": {"t": "taste", "a": "aroma", "j": "juiciness", "n": "nutrition", "m": "market"},
    "Lung": {"g": "gender", "a": "age", "s": "smoking", "l": "lesion"},
}


def pairmask_view(dataset, text, code_a, code_b):
    """Text view that removes TWO categories' keywords (keeps the other 3)."""
    kw, code = _KW[dataset], _CAT_CODE[dataset]
    remove = kw[code[code_a]] | kw[code[code_b]]
    return _mask_keywords(text, remove) or text


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


def mag_category_view(review, category):
    """MAG text view for one category: remove all OTHER categories' keywords,
    keeping this category's vocabulary. Falls back to original if emptied."""
    others = set()
    for c, kw in MAG_KW.items():
        if c != category:
            others |= kw
    return _mask_keywords(review, others) or review


# --- driver ---------------------------------------------------------------


def _save(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


_IMG_OPS = {"img_color": color_view, "img_hpf": hpf_view, "img_silhouette": silhouette_view}


def _base_text(dataset, row):
    """The full text a dataset's pair-mask views are derived from."""
    if dataset == "MAG":
        return str(row["Review"])
    if dataset == "Lung":                            # concat Demographics + History
        return f"{str(row['Demographics']).strip()} {str(row['History']).strip()}".strip()
    raise ValueError(dataset)


def _text_views(dataset, row):
    """Return {view_name: text} for the pair-mask text views of `dataset`."""
    text = _base_text(dataset, row)
    out = {}
    for name in C.views(dataset):
        if name.startswith("txt_pm_"):
            a, b = name.split("_")[-1]               # e.g. "rs" -> ('r','s')
            out[name] = pairmask_view(dataset, text, a, b)
    return out


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
        row.update(_text_views(dataset, meta.iloc[i]))
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
