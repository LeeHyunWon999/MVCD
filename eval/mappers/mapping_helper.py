"""Prepare and apply compact human review files for dataset-wide GT mappers.

Workflow (run from the project root):
  python eval/mappers/mapping_helper.py init MAG
  # Edit eval/mappers/MAG_unreviewed_helper.json:
  #   null       = not reviewed yet
  #   []         = nulled/spurious
  #   [2]        = one GT
  #   [4, 5]     = primary GT first, remaining GTs become compound.secondary
  python eval/mappers/mapping_helper.py check MAG
  python eval/mappers/mapping_helper.py apply MAG

Use --mapper-file and --review-file for an isolated run-specific mapper.
"""
import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from eval.gt_mapper import GT_NODES, load_mapper, mapper_path, validate_mapper
from mvext.artifacts import write_json


HELPER_ROOT = Path(__file__).resolve().parent
GT_CODES = {
    "Lung": {1: "lesion", 2: "age", 3: "gender", 4: "smoking"},
    "MAG": {1: "defects", 2: "color", 3: "size", 4: "aroma",
            5: "taste", 6: "nutrition", 7: "recmd", 8: "juiciness"},
}


def helper_path(dataset):
    return HELPER_ROOT / f"{dataset}_unreviewed_helper.json"


def init_review(dataset, mapper_file=None, review_file=None):
    mapper = load_mapper(dataset, mapper_file)
    path = Path(review_file or helper_path(dataset))
    existing = json.loads(path.read_text()) if path.exists() else {}
    if not isinstance(existing, dict):
        raise ValueError(f"helper JSON must be an object: {path}")
    review = {name: existing.get(name) for name in mapper["unreviewed"]}
    write_json(path, review)
    return path, review


def _targets(dataset, name, value):
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"{name!r}: value must be null or a list")
    targets = []
    for token in value:
        if isinstance(token, int) and not isinstance(token, bool):
            target = GT_CODES[dataset].get(token)
        elif isinstance(token, str) and token.strip():
            target = token.strip().casefold()
        else:
            raise ValueError(f"{name!r}: GT entries must be integer codes or names")
        if target not in GT_NODES[dataset]:
            raise ValueError(f"{name!r}: invalid GT {token!r}")
        targets.append(target)
    if len(targets) != len(set(targets)):
        raise ValueError(f"{name!r}: duplicate GT after alias expansion")
    return targets


def check_review(dataset, mapper_file=None, review_file=None):
    mapper = load_mapper(dataset, mapper_file)
    path = Path(review_file or helper_path(dataset))
    review = json.loads(path.read_text())
    if not isinstance(review, dict):
        raise ValueError(f"helper JSON must be an object: {path}")
    expected, actual = set(mapper["unreviewed"]), set(review)
    if expected != actual:
        missing, extra = sorted(expected - actual), sorted(actual - expected)
        raise ValueError(
            f"helper keys differ from mapper; missing={missing[:5]}, extra={extra[:5]}")
    assignments, invalid = {}, []
    for name, value in review.items():
        try:
            assignments[name] = _targets(dataset, name, value)
        except ValueError as error:
            invalid.append(str(error))
    if invalid:
        details = "\n  - ".join(invalid[:20])
        raise ValueError(f"{len(invalid)} invalid assignments:\n  - {details}")
    incomplete = [name for name, targets in assignments.items() if targets is None]
    return assignments, incomplete


def apply_review(dataset, mapper_file=None, review_file=None):
    path = Path(mapper_file or mapper_path(dataset))
    mapper = load_mapper(dataset, path)
    assignments, incomplete = check_review(dataset, path, review_file)
    if incomplete:
        raise ValueError(f"{len(incomplete)} factors are still null; first={incomplete[0]!r}")

    stats = {"mapped": 0, "compound": 0, "nulled": 0}
    for name, targets in assignments.items():
        runs = list(mapper["unreviewed"][name]["runs"])
        if not targets:
            mapper["nulled"][name] = runs
            stats["nulled"] += 1
        else:
            primary = targets[0]
            mapper["gt_factors"][primary][name] = runs
            stats["mapped"] += 1
            if len(targets) > 1:
                mapper["compound"][name] = {
                    "primary": primary, "secondary": targets[1:], "runs": list(runs)}
                stats["compound"] += 1
        del mapper["unreviewed"][name]

    errors = validate_mapper(mapper, require_reviewed=True)
    if errors:
        raise ValueError("invalid mapper after helper apply: " + "; ".join(errors))
    write_json(path, mapper)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["init", "check", "apply"])
    parser.add_argument("dataset", choices=["MAG", "Lung"])
    parser.add_argument("--mapper-file")
    parser.add_argument("--review-file")
    args = parser.parse_args()

    if args.command == "init":
        path, review = init_review(args.dataset, args.mapper_file, args.review_file)
        done = sum(value is not None for value in review.values())
        print(f"[{args.dataset}] {path}: total={len(review)} done={done} "
              f"remaining={len(review) - done}")
        return

    assignments, incomplete = check_review(
        args.dataset, args.mapper_file, args.review_file)
    if args.command == "check":
        print(f"[{args.dataset}] total={len(assignments)} "
              f"complete={len(assignments) - len(incomplete)} remaining={len(incomplete)}")
        if incomplete:
            print("first remaining:")
            for name in incomplete[:20]:
                print(f"  - {name}")
            raise SystemExit(2)
        return

    if incomplete:
        raise SystemExit(f"{len(incomplete)} factors are still null; run check for details")
    stats = apply_review(args.dataset, args.mapper_file, args.review_file)
    print(f"[{args.dataset}] applied: {stats}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        raise SystemExit(f"ERROR: {error}")
