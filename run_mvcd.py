"""Run the paper MVCD pipeline."""
import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs" / "mvcd.py"


def run(command, env):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("MAG", "Lung", "both"), default="both")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage",
                        choices=("prepare", "train", "name", "graph", "evaluate", "all"),
                        default="all")
    parser.add_argument("--output-root", default="outputs/mvcd")
    args = parser.parse_args()

    datasets = ("MAG", "Lung") if args.dataset == "both" else (args.dataset,)
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    block_root = output_root / "blocks" / f"seed{args.seed:02d}"
    env = {
        **os.environ,
        "MVEXT_CONFIG": str(CONFIG),
        "MVEXT_RUN_SEED": str(args.seed),
        "PYTHONHASHSEED": str(args.seed),
        "MODEL": os.environ.get("MODEL", "gemini-3.1-flash-image"),
    }

    if args.stage in ("name", "graph", "all") and not env.get("API_KEY"):
        raise SystemExit("API_KEY is required for factor naming and quantification")
    if args.stage in ("prepare", "all"):
        dataset = "both" if len(datasets) == 2 else datasets[0]
        run([sys.executable, "-m", "mvext.augment", "--dataset", dataset], env)
        run([sys.executable, "-m", "mvext.encoders", "--dataset", dataset], env)
    if args.stage in ("train", "all"):
        dataset = "both" if len(datasets) == 2 else datasets[0]
        run([sys.executable, "-m", "mvext.validity", "--dataset", dataset,
             "--run-seed", str(args.seed), "--output-root", str(block_root)], env)
    if args.stage in ("name", "all"):
        for dataset in datasets:
            blocks = block_root / dataset / "blocks.pt"
            if not blocks.exists():
                raise SystemExit(f"missing block artifact: {blocks}")
            run_env = {**env, "MVEXT_RUN_TAG":
                       f"mvcd_{dataset}_seed{args.seed:02d}"}
            run([sys.executable, "-m", "mvext.main_mv", "--dataset", dataset,
                 "--nodes-only", "--run-seed", str(args.seed),
                 "--data-root", str(block_root)], run_env)
    if args.stage in ("graph", "all"):
        for dataset in datasets:
            blocks = block_root / dataset / "blocks.pt"
            run_id = f"mvcd_{dataset}_seed{args.seed:02d}"
            run_dir = ROOT / "outputs" / ("MAG9_mv" if dataset == "MAG" else "Lung4_mv") / run_id
            if not blocks.exists():
                raise SystemExit(f"missing block artifact: {blocks}")
            if not (run_dir / "iter1_factors.json").exists():
                raise SystemExit(f"run the naming stage first: {run_dir}")
            run_env = {**env, "MVEXT_RUN_TAG": run_id}
            run([sys.executable, "-m", "mvext.main_mv", "--dataset", dataset,
                 "--iter1-only", "--quantify-with", "llm",
                 "--run-seed", str(args.seed), "--data-root", str(block_root)], run_env)
    if args.stage == "evaluate":
        for dataset in datasets:
            run([sys.executable, "-m", "eval.paper_metrics", dataset,
                 f"mvcd_{dataset}_seed{args.seed:02d}"], env)


if __name__ == "__main__":
    main()
