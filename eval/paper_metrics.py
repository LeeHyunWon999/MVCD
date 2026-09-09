"""Paper metrics against the dataset ground-truth graph."""
import argparse
import json
from pathlib import Path

from causallearn.graph.AdjacencyConfusion import AdjacencyConfusion
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.SHD import SHD

from eval.graph_io import load_graph_json
from eval.ground_truth import GROUND_TRUTH
from eval.gt_mapper import export_alignment
from mvext.artifacts import write_json


OUTPUT_ROOT = {"MAG": Path("outputs/MAG9_mv"), "Lung": Path("outputs/Lung4_mv")}
ENDPOINTS = {name: getattr(Endpoint, name) for name in ("TAIL", "ARROW", "CIRCLE")}


def _run_path(dataset, run):
    path = Path(run)
    return path if path.is_dir() else OUTPUT_ROOT[dataset] / path


def _causal_graph(graph, names, alignment=None):
    """Project a stored graph onto `names`, matching the authors' evaluator."""
    names = sorted(set(names))
    nodes = {name: GraphNode(name) for name in names}
    result = GeneralGraph(list(nodes.values()))
    added = set()
    for edge in graph["edges"]:
        left = alignment.get(edge["from"]) if alignment is not None else edge["from"]
        right = alignment.get(edge["to"]) if alignment is not None else edge["to"]
        if left not in nodes or right not in nodes or left == right:
            continue
        signature = frozenset((left, right))
        if signature in added:
            continue
        result.add_edge(Edge(nodes[left], nodes[right],
                             ENDPOINTS[edge["ep_from"]], ENDPOINTS[edge["ep_to"]]))
        added.add(signature)
    return result


def evaluate(dataset, predicted, alignment, node_penalty=1):
    """Return NP/NR/NF/AP/AR/AF/ESHD using the MLLM-CD definitions."""
    ground_truth = GROUND_TRUTH[dataset]
    gt_nodes = set(ground_truth["nodes"])
    pred_nodes = set(predicted["nodes"])
    mapped = {target for name in pred_nodes
              if (target := alignment.get(name)) in gt_nodes}
    matched = gt_nodes & mapped
    missing = gt_nodes - matched
    spurious = {name for name in pred_nodes if alignment.get(name) is None}

    np_ = len(matched) / len(pred_nodes) if pred_nodes else 0.0
    nr = len(matched) / len(gt_nodes) if gt_nodes else 0.0
    nf = 2 * np_ * nr / (np_ + nr) if np_ + nr else 0.0

    gt_subgraph = _causal_graph(ground_truth, matched)
    pred_subgraph = _causal_graph(predicted, matched, alignment)
    confusion = AdjacencyConfusion(gt_subgraph, pred_subgraph)
    tp, fp, fn = (confusion.get_adj_tp(), confusion.get_adj_fp(),
                  confusion.get_adj_fn())
    ap = tp / (tp + fp) if tp + fp else 0.0
    ar = tp / (tp + fn) if tp + fn else 0.0
    af = 2 * ap * ar / (ap + ar) if ap + ar else 0.0
    shd = SHD(gt_subgraph, pred_subgraph).get_shd() if matched else 0

    missing_edges = sum(edge["from"] in missing or edge["to"] in missing
                        for edge in ground_truth["edges"])
    spurious_edges = sum(edge["from"] in spurious or edge["to"] in spurious
                         for edge in predicted["edges"])
    eshd = (shd + missing_edges + spurious_edges
            + node_penalty * (len(missing) + len(spurious)))
    return {"NP": round(np_, 4), "NR": round(nr, 4), "NF": round(nf, 4),
            "AP": round(ap, 4), "AR": round(ar, 4), "AF": round(af, 4),
            "ESHD": int(eshd)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("MAG", "Lung"))
    parser.add_argument("run", help="run directory or run ID under outputs/<dataset>")
    args = parser.parse_args()
    run = _run_path(args.dataset, args.run)
    graph_path = run / "graph_iter1.json"
    if not graph_path.exists():
        raise SystemExit(f"missing FCI graph; run the graph stage first: {graph_path}")
    alignment, diagnostics = export_alignment(args.dataset, run)
    metrics = evaluate(args.dataset, load_graph_json(graph_path), alignment)
    result = {"schema_version": 1, "dataset": args.dataset,
              "reference": "ground_truth", "metrics": metrics,
              "diagnostics": diagnostics}
    write_json(run / "metrics_iter1.json", result)
    print(json.dumps({**metrics, **diagnostics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
