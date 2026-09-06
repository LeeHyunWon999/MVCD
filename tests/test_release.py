"""Small release contract check; run directly with Python."""
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["MVEXT_CONFIG"] = str(ROOT / "configs" / "run4_selected.py")

from eval.gt_mapper import export_alignment, new_mapper, sync_run, validate_mapper  # noqa: E402
from eval.node_metrics import evaluate_nodes  # noqa: E402
from mvext import config as C  # noqa: E402
from mvext.artifacts import write_json  # noqa: E402
from mvext.naming import discover_factors_from_blocks  # noqa: E402


assert C.CONTENT_SIZE_MODE == "variable"
assert C.LOSS == "infonce" and C.LOSS_SIM == "l2"
assert C.VALIDITY_GATE_ENABLED is False
assert C.DECOMPOSE_ENABLED is C.RANK_NORMALIZE is C.DEDUP_ENABLED is False
assert C.NAMING_USE_LOSS_SIM is False
assert C.TOP_N == 10 and C.LAMBDA_AVERAGE is True
assert C.QUANT == "multiview" and C.NODES_ONLY is True

for dataset in ("MAG", "Lung"):
    mapper = json.loads((ROOT / "eval" / "mappers" / f"{dataset}.json").read_text())
    assert not validate_mapper(mapper)
    buckets = [*mapper["gt_factors"].values(), mapper["nulled"]]
    assert all(not runs for bucket in buckets for runs in bucket.values())
    assert all(not entry["runs"] for entry in mapper["compound"].values())

assert evaluate_nodes("Lung", {
    "score": "diagnosis", "Patient Age": "age", "Artifact": None,
}) == {"NP": 0.6667, "NR": 0.4, "NF": 0.5}


class FakeHelper:
    def generate(self, prompt, img_path=None):
        assert "Score" in prompt and len(img_path) == 2
        return {"text": "fake"}

    def extract_factors_from_response(self, response):
        assert response == "fake"
        return [{"name": "Synthetic Factor", "criteria": {"neutral": "otherwise"}}]


meta = pd.DataFrame({
    "score": [0, 1], "Review": ["first", "second"],
    "ImagePath": [str(ROOT / "Lung" / "1.jpg"), str(ROOT / "Lung" / "2.jpg")],
})
factors, pairs = discover_factors_from_blocks(
    FakeHelper(), meta,
    {"block_values": [torch.tensor([[0.0], [1.0]])], "block_subsets": [(0, 1)]},
    "Lung", top_n=1, quant="multiview")
assert factors[0]["name"] == "Synthetic Factor" and pairs[(0, 1)][0][:2] == (0, 1)

with tempfile.TemporaryDirectory() as temporary:
    temporary = Path(temporary)
    run, mapper_path = temporary / "run4_selected_Lung_seed00", temporary / "Lung.json"
    write_json(run / "iter1_factors.json", [{"name": "Patient Age", "criteria": {}}])
    mapper = new_mapper("Lung")
    mapper["gt_factors"]["diagnosis"]["score"] = []
    mapper["gt_factors"]["age"]["Patient Age"] = []
    write_json(mapper_path, mapper)
    sync_run("Lung", run, mapper_path)
    alignment, diagnostics = export_alignment("Lung", run, mapper_path)
    assert alignment == {"score": "diagnosis", "Patient Age": "age"}
    assert diagnostics["Predicted Factor Count"] == diagnostics["Matched GT Factor Count"] == 1

with (ROOT / "MAG9.csv").open() as stream:
    import csv
    assert all((ROOT / row["ImagePath"]).is_file() for row in csv.DictReader(stream))
assert len(list((ROOT / "Lung").glob("*.jpg"))) == 60

print("release checks passed")
