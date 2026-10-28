"""Public entry point for CDAF training, evaluation, and inference."""

import argparse
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent


def build_parser():
    parser = argparse.ArgumentParser(
        description="CDAF: context-conditioned diffusion and availability-aware evidence fusion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["train", "evaluate", "predict", "verify-data"], default="train"
    )
    parser.add_argument(
        "--dataset", choices=["B-dataset", "C-dataset", "F-dataset", "all"], default="all"
    )
    parser.add_argument(
        "--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="Fold indices to run"
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument(
        "--weights-root", type=Path, help="Training run directory used by evaluate mode"
    )
    parser.add_argument(
        "--run-name", help="Optional output folder name; existing folders are never overwritten"
    )
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, or cuda:N; one device per process"
    )
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--eval-split", choices=["val", "test"], default="val")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Train one batch per stage and one epoch; not paper results",
    )
    for name in ("epochs", "warmup-epochs", "diffusion-epochs", "hidden-dim"):
        parser.add_argument("--" + name, type=int, help="Override the dataset JSON configuration")
    for name in ("finetune-lr", "lambda-diffusion", "lambda-aux"):
        parser.add_argument("--" + name, type=float, help="Override the dataset JSON configuration")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint for predict mode")
    parser.add_argument(
        "--pairs-file", type=Path, help="CSV with zero-based integer drug,protein columns"
    )
    parser.add_argument("--prediction-output", type=Path, default=Path("predictions.csv"))
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if (
        not args.folds
        or len(set(args.folds)) != len(args.folds)
        or any(f not in range(5) for f in args.folds)
    ):
        parser.error("--folds must contain distinct indices in 0..4")
    if args.num_threads < 1:
        parser.error("--num-threads must be positive")
    if args.mode == "predict" and (args.checkpoint is None or args.pairs_file is None):
        parser.error("predict requires --checkpoint and --pairs-file")
    if args.mode == "evaluate" and args.weights_root is None:
        parser.error("evaluate requires --weights-root pointing to a completed training run")
    if args.run_name is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_name):
        parser.error(
            "--run-name must start with a letter or digit and use only letters, digits, _, -, or ."
        )
    if args.smoke and args.mode != "train":
        parser.error("--smoke applies to train mode only")
    from cdaf.release_runner import run

    run(args)


if __name__ == "__main__":
    main()
