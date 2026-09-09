"""Dataset-wide factor-name mapper for MVCD evaluation."""
import argparse
import json
from pathlib import Path

from eval.graph_io import load_graph_json
from eval.ground_truth import LUNG_NODES, MAG_NODES
from mvext.artifacts import write_json


GT_NODES = {"MAG": MAG_NODES, "Lung": LUNG_NODES}
TARGET_GT = {"MAG": "score", "Lung": "diagnosis"}
OUTPUT_ROOT = {"MAG": Path("outputs/MAG9_mv"), "Lung": Path("outputs/Lung4_mv")}
MAPPER_ROOT = Path("eval/mappers")
NULL_LABEL = "__null__"


def mapper_path(dataset):
    return MAPPER_ROOT / f"{dataset}.json"


def new_mapper(dataset):
    return {
        "schema_version": 1,
        "dataset": dataset,
        "gt_factors": {name: {} for name in GT_NODES[dataset]},
        "compound": {},
        "nulled": {},
        "unreviewed": {},
    }


def load_mapper(dataset, path=None):
    path = Path(path or mapper_path(dataset))
    return json.loads(path.read_text()) if path.exists() else new_mapper(dataset)


def _add_run(runs, run_id):
    if run_id not in runs:
        runs.append(run_id)
        runs.sort()


def _unreviewed_entry(mapper, name):
    entry = mapper["unreviewed"].setdefault(name, {"runs": [], "observed": {}})
    if isinstance(entry, list):
        entry = {"runs": entry, "observed": {}}
        mapper["unreviewed"][name] = entry
    return entry


def _classification(mapper):
    found = {}
    for gt, names in mapper["gt_factors"].items():
        for name in names:
            found.setdefault(name, []).append(("gt", gt))
    for name in mapper["nulled"]:
        found.setdefault(name, []).append(("nulled", None))
    for name in mapper["unreviewed"]:
        found.setdefault(name, []).append(("unreviewed", None))
    return found


def _resolve_run(dataset, run):
    path = Path(run)
    if path.is_dir():
        return path
    path = OUTPUT_ROOT[dataset] / run
    if not path.is_dir():
        raise FileNotFoundError(f"run folder not found: {path}")
    return path


def _run_names(run):
    factor_path = run / "iter1_factors.json"
    if factor_path.exists():
        names = [factor["name"] for factor in json.loads(factor_path.read_text())]
        return ["score", *names]
    graphs = sorted(run.glob("*/graph_iter1_*.json"))
    if not graphs:
        raise FileNotFoundError(f"no iter1 factors or graph under {run}")
    return load_graph_json(str(graphs[-1]))["nodes"]


def validate_mapper(mapper, require_reviewed=True):
    errors = []
    dataset = mapper.get("dataset")
    if dataset not in GT_NODES:
        return [f"invalid dataset: {dataset!r}"]
    if set(mapper.get("gt_factors", {})) != set(GT_NODES[dataset]):
        errors.append("gt_factors keys do not match canonical GT nodes")

    found = _classification(mapper)
    for name, locations in found.items():
        if len(locations) != 1:
            errors.append(f"{name!r} appears in multiple classification buckets: {locations}")

    def check_runs(label, runs):
        if not isinstance(runs, list):
            errors.append(f"{label} runs must be a list")
        elif runs != sorted(set(runs)):
            errors.append(f"{label} runs must be sorted and unique")

    for gt, names in mapper.get("gt_factors", {}).items():
        for name, runs in names.items():
            check_runs(f"gt_factors.{gt}.{name}", runs)
    for name, runs in mapper.get("nulled", {}).items():
        check_runs(f"nulled.{name}", runs)
    for name, entry in mapper.get("unreviewed", {}).items():
        if not isinstance(entry, dict):
            errors.append(f"unreviewed.{name} must be an object")
            continue
        check_runs(f"unreviewed.{name}", entry.get("runs"))
        for observed, runs in entry.get("observed", {}).items():
            if observed != NULL_LABEL and observed not in GT_NODES[dataset]:
                errors.append(f"unreviewed.{name} has invalid observed target {observed!r}")
            check_runs(f"unreviewed.{name}.observed.{observed}", runs)

    for name, entry in mapper.get("compound", {}).items():
        if not isinstance(entry, dict):
            errors.append(f"compound.{name} must be an object")
            continue
        primary = entry.get("primary")
        if primary not in GT_NODES[dataset]:
            errors.append(f"compound.{name} has invalid primary {primary!r}")
            continue
        if "secondary" in entry:
            secondary = entry["secondary"]
            if not isinstance(secondary, list) or not secondary:
                errors.append(f"compound.{name} secondary must be a non-empty list")
            elif not all(isinstance(target, str) for target in secondary):
                errors.append(f"compound.{name} secondary entries must be strings")
            else:
                if len(secondary) != len(set(secondary)):
                    errors.append(f"compound.{name} secondary must be unique")
                for target in secondary:
                    if target not in GT_NODES[dataset]:
                        errors.append(
                            f"compound.{name} has invalid secondary {target!r}")
                    elif target == primary:
                        errors.append(
                            f"compound.{name} repeats primary in secondary")
        primary_runs = mapper["gt_factors"].get(primary, {}).get(name)
        if primary_runs is None:
            errors.append(f"compound.{name} is missing from gt_factors.{primary}")
            continue
        check_runs(f"compound.{name}", entry.get("runs"))
        if entry.get("runs") != primary_runs:
            errors.append(f"compound.{name} runs differ from gt_factors.{primary}")

    if require_reviewed and mapper.get("unreviewed"):
        errors.append(f"{len(mapper['unreviewed'])} unreviewed factor names remain")
    return errors


def sync_run(dataset, run, path=None):
    path = Path(path or mapper_path(dataset))
    mapper = load_mapper(dataset, path)
    run = _resolve_run(dataset, run)
    run_id = run.name
    classified = _classification(mapper)

    for name in _run_names(run):
        locations = classified.get(name, [])
        if not locations:
            entry = _unreviewed_entry(mapper, name)
            _add_run(entry["runs"], run_id)
            classified[name] = [("unreviewed", None)]
            continue
        if len(locations) != 1:
            raise ValueError(f"{name!r} has ambiguous mapper locations: {locations}")
        bucket, target = locations[0]
        if bucket == "gt":
            _add_run(mapper["gt_factors"][target][name], run_id)
            if name in mapper["compound"]:
                _add_run(mapper["compound"][name]["runs"], run_id)
        elif bucket == "nulled":
            _add_run(mapper["nulled"][name], run_id)
        else:
            _add_run(_unreviewed_entry(mapper, name)["runs"], run_id)

    errors = validate_mapper(mapper, require_reviewed=False)
    if errors:
        raise ValueError("invalid mapper after sync: " + "; ".join(errors))
    write_json(path, mapper)
    return mapper


def resolve_name(dataset, name, target, path=None, compound=False):
    path = Path(path or mapper_path(dataset))
    mapper = load_mapper(dataset, path)
    if name not in mapper["unreviewed"]:
        raise KeyError(f"factor is not unreviewed: {name!r}")
    entry = _unreviewed_entry(mapper, name)
    runs = entry["runs"]
    if target in (None, "null", NULL_LABEL):
        if compound:
            raise ValueError("a nulled factor cannot be compound")
        mapper["nulled"][name] = runs
    else:
        if target not in GT_NODES[dataset]:
            raise ValueError(f"invalid GT target: {target!r}")
        mapper["gt_factors"][target][name] = runs
        if compound:
            mapper["compound"][name] = {"primary": target, "runs": list(runs)}
    del mapper["unreviewed"][name]
    errors = validate_mapper(mapper, require_reviewed=False)
    if errors:
        raise ValueError("invalid mapper after resolve: " + "; ".join(errors))
    write_json(path, mapper)
    return mapper


def export_alignment(dataset, run, path=None):
    mapper = load_mapper(dataset, path)
    errors = validate_mapper(mapper, require_reviewed=True)
    if errors:
        raise ValueError("mapper is not ready: " + "; ".join(errors))
    run = _resolve_run(dataset, run)
    names = _run_names(run)
    classified = _classification(mapper)
    alignment = {}
    for name in names:
        locations = classified.get(name, [])
        if len(locations) != 1 or locations[0][0] == "unreviewed":
            raise ValueError(f"factor is not reviewed: {name!r}")
        bucket, target = locations[0]
        alignment[name] = target if bucket == "gt" else None

    predicted = [name for name in names if name != "score"]
    matched = sorted({alignment[name] for name in predicted if alignment[name] is not None
                      and alignment[name] != TARGET_GT[dataset]})
    compound = sorted(name for name in predicted if name in mapper["compound"])
    diagnostics = {
        "Predicted Factor Count": len(predicted),
        "Matched GT Factor Count": len(matched),
        "Matched GT Factors": matched,
        "Multi-GT Factor Count": len(compound),
        "Multi-GT Factor Names": compound,
    }
    write_json(run / "alignment_iter1.json", alignment)
    write_json(run / "node_diagnostics.json", diagnostics)
    return alignment, diagnostics


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sync = sub.add_parser("sync")
    sync.add_argument("dataset", choices=["MAG", "Lung"])
    sync.add_argument("runs", nargs="+")
    validate = sub.add_parser("validate")
    validate.add_argument("dataset", choices=["MAG", "Lung"])
    validate.add_argument("--allow-unreviewed", action="store_true")
    resolve = sub.add_parser("resolve")
    resolve.add_argument("dataset", choices=["MAG", "Lung"])
    resolve.add_argument("name")
    resolve.add_argument("target", help="canonical GT name or 'null'")
    resolve.add_argument("--compound", action="store_true")
    export = sub.add_parser("export")
    export.add_argument("dataset", choices=["MAG", "Lung"])
    export.add_argument("run")
    args = parser.parse_args()

    if args.command == "sync":
        for run in args.runs:
            sync_run(args.dataset, run)
            print(f"[{args.dataset}] synced {run}")
    elif args.command == "validate":
        errors = validate_mapper(load_mapper(args.dataset), not args.allow_unreviewed)
        if errors:
            raise SystemExit("\n".join(f"- {error}" for error in errors))
        print(f"[{args.dataset}] mapper valid")
    elif args.command == "resolve":
        mapper = resolve_name(args.dataset, args.name, args.target,
                              compound=args.compound)
        print(f"[{args.dataset}] resolved {args.name!r} -> {args.target}; "
              f"remaining={len(mapper['unreviewed'])}")
    else:
        _, diagnostics = export_alignment(args.dataset, args.run)
        print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
