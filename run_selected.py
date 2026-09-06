"""Run the selected run4 node-discovery configuration."""
import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs" / "run4_selected.py"


def run(command, env):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("MAG", "Lung", "both"), default="both")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage", choices=("prepare", "train", "name", "all"),
                        default="all")
    parser.add_argument("--output-root", default="outputs/run4_selected")
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

    if args.stage in ("name", "all") and not env.get("API_KEY"):
        raise SystemExit("API_KEY is required for factor naming")
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
                       f"run4_selected_{dataset}_seed{args.seed:02d}"}
            run([sys.executable, "-m", "mvext.main_mv", "--dataset", dataset,
                 "--nodes-only", "--run-seed", str(args.seed),
                 "--data-root", str(block_root)], run_env)


if __name__ == "__main__":
    main()
