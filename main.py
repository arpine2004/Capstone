"""
main.py 

Executes all project notebooks in the correct order.
Each notebook loads from existing checkpoints. No training from scratch.

Usage
-----
    python main.py                  # run all notebooks
    python main.py --only EDA LDM  # run a named subset
    python main.py --skip LDM      # skip specific notebooks

Notebooks are executed in-place. Outputs are saved back to the .ipynb files.
A timestamped execution log is written to reports/run_<timestamp>.log.
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# Notebook execution order

NOTEBOOKS = [
    ("EDA", "notebooks/EDA.ipynb"),
    ("AlexNet", "notebooks/AlexNet.ipynb"),
    ("VGG16", "notebooks/VGG16.ipynb"),
    ("ViT", "notebooks/ViT.ipynb"),
    ("ResNet101", "notebooks/ResNet101.ipynb"),
    ("Models", "notebooks/Models.ipynb"),
    ("FusionModelCrossAttn", "notebooks/FusionModelCrossAttn.ipynb"),
    ("PhysicallyInformedLateFusion", "notebooks/PhysicallyInformedLateFusion.ipynb"),
    ("Model_Comparison", "notebooks/Model_Comparison.ipynb"),
    ("GalaxyLDM", "notebooks/GalaxyLDM.ipynb"),
]


# Setup
ROOT = Path(__file__).parent.resolve()


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("capstone")


def run_notebook(name: str, rel_path: str, log: logging.Logger) -> bool:
    """Execute a single notebook via nbconvert. Returns True on success."""
    nb_path = ROOT / rel_path
    if not nb_path.exists():
        log.error(f"[{name}] Not found: {nb_path}")
        return False

    log.info(f"[{name}] Starting — {rel_path}")
    t0 = time.time()

    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.timeout=-1",      
        "--ExecutePreprocessor.kernel_name=python3",
        str(nb_path),
    ]

    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )

    elapsed = time.time() - t0
    mins, secs = divmod(int(elapsed), 60)

    if result.returncode == 0:
        log.info(f"[{name}] Done in {mins}m {secs}s")
        return True
    else:
        log.error(f"[{name}] FAILED after {mins}m {secs}s")

        stderr_tail = "\n".join(result.stderr.strip().splitlines()[-40:])
        log.error(f"[{name}] stderr:\n{stderr_tail}")
        return False


# CLI
def parse_args():
    parser = argparse.ArgumentParser(description="Run Capstone notebooks in order.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--only", nargs="+", metavar="NAME",
        help="Run only these notebooks (by short name, e.g. EDA AlexNet LDM).",
    )
    group.add_argument(
        "--skip", nargs="+", metavar="NAME",
        help="Skip these notebooks and run the rest.",
    )
    return parser.parse_args()


def resolve_notebooks(args) -> list[tuple[str, str]]:
    """Filter NOTEBOOKS according to --only / --skip flags."""
    if args.only:
        names = {n.upper() for n in args.only}
        subset = [(n, p) for n, p in NOTEBOOKS if n.upper() in names or
                  any(alias in n.upper() for alias in names)]
        if not subset:
            print(f"No matching notebooks for --only {args.only}. "
                  f"Available: {[n for n, _ in NOTEBOOKS]}")
            sys.exit(1)
        return subset
    if args.skip:
        names = {n.upper() for n in args.skip}
        return [(n, p) for n, p in NOTEBOOKS if n.upper() not in names and
                not any(alias in n.upper() for alias in names)]
    return NOTEBOOKS



# Main
def main():
    args = parse_args()
    selected = resolve_notebooks(args)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = ROOT / "reports" / f"run_{timestamp}.log"
    log = setup_logging(log_path)

    log.info("=" * 60)
    log.info("Capstone — reproducibility run")
    log.info(f"Notebooks to run: {[n for n, _ in selected]}")
    log.info(f"Log: {log_path}")
    log.info("=" * 60)

    results = {}
    run_t0 = time.time()

    for name, path in selected:
        ok = run_notebook(name, path, log)
        results[name] = ok
        if not ok:
            log.warning(f"[{name}] Failed — continuing with remaining notebooks.")

    total_elapsed = time.time() - run_t0
    total_mins, total_secs = divmod(int(total_elapsed), 60)

    log.info("=" * 60)
    log.info(f"Total time: {total_mins}m {total_secs}s")
    log.info("Results:")
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        log.info(f"  {name:<35} {status}")

    failed = [n for n, ok in results.items() if not ok]
    if failed:
        log.error(f"{len(failed)} notebook(s) failed: {failed}")
        sys.exit(1)
    else:
        log.info("All notebooks completed successfully.")


if __name__ == "__main__":
    main()
