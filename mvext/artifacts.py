"""Small artifact helpers shared by MVCD entry points."""
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)
    return path


def write_graph_png(graph, path):
    """Write an optional graph preview without making Graphviz a run dependency."""
    if shutil.which("dot") is None:
        print(f"Warning: Graphviz 'dot' not found; skipped PNG: {path}")
        return False
    graph.write_png(path)
    print(f"Causal graph visualization saved: {path}")
    return True


def save_original_iter1(output_path, dataset, run_seed, iter1_only, factors, started_at,
                        status=None):
    import gemini_utils
    output_path = Path(output_path)
    factors_path = write_json(output_path / "iter1_factors.json", factors)
    run_id = os.environ.get("MVEXT_RUN_TAG") or output_path.name
    status = status or ("complete" if iter1_only else "iter1_complete")
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "pipeline": "original_mllm_cd",
        "dataset": dataset,
        "run_seed": int(run_seed),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "iter1_only": bool(iter1_only),
        "model": os.environ.get("MODEL"),
        "generation_config": gemini_utils.generation_config_dict(),
        "status": status,
        "started_at": started_at,
        "iter1_completed_at": utc_now(),
        "factor_count": len(factors),
        "files": {"factors": factors_path.name},
    }
    if status == "complete":
        manifest["completed_at"] = utc_now()
    write_json(output_path / "manifest.json", manifest)
    return manifest
