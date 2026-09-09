"""Canonical ground-truth graphs used by the paper metrics."""

from eval.graph_io import directed_graph_dict

MAG_NODES = ["color", "size", "defects", "aroma", "taste",
             "juiciness", "nutrition", "recmd", "score"]
LUNG_NODES = ["gender", "age", "smoking", "lesion", "diagnosis"]

MAG_EDGES = [
    ("color", "score"), ("size", "score"), ("defects", "aroma"),
    ("aroma", "score"), ("defects", "taste"), ("taste", "score"),
    ("taste", "juiciness"), ("score", "recmd"), ("nutrition", "recmd"),
]
LUNG_EDGES = [
    ("gender", "diagnosis"), ("age", "diagnosis"),
    ("smoking", "diagnosis"), ("lesion", "diagnosis"),
]

GROUND_TRUTH = {
    "MAG": directed_graph_dict(MAG_NODES, MAG_EDGES),
    "Lung": directed_graph_dict(LUNG_NODES, LUNG_EDGES),
}
