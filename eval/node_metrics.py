"""NP/NR/NF evaluation for a reviewed node-only run."""
import argparse
import json
from pathlib import Path

from eval.ground_truth import LUNG_NODES, MAG_NODES
from eval.gt_mapper import export_alignment
from mvext.artifacts import write_json


GT_NODES = {"MAG": set(MAG_NODES), "Lung": set(LUNG_NODES)}


def evaluate_nodes(dataset, alignment):
    predicted = set(alignment)
    matched = GT_NODES[dataset].intersection(
        target for target in alignment.values() if target is not None)
    precision = len(matched) / len(predicted) if predicted else 0.0
    recall = len(matched) / len(GT_NODES[dataset])
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"NP": round(precision, 4), "NR": round(recall, 4), "NF": round(f1, 4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("MAG", "Lung"))
    parser.add_argument("run", help="run directory or run ID under outputs/<dataset>")
    args = parser.parse_args()
    alignment, diagnostics = export_alignment(args.dataset, args.run)
    metrics = evaluate_nodes(args.dataset, alignment)
    run = Path(args.run)
    if not run.is_dir():
        run = Path("outputs") / ("MAG9_mv" if args.dataset == "MAG" else "Lung4_mv") / args.run
    write_json(run / "node_metrics.json", {
        "schema_version": 1, "dataset": args.dataset,
        "metrics": metrics, "diagnostics": diagnostics,
    })
    print(json.dumps({**metrics, **diagnostics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
