"""Configuration schema for the fixed CDAF architecture and its ablations."""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_parser():
    """Return internal model/training defaults; main.py is the public entry point."""
    parser = argparse.ArgumentParser(description="CDAF configuration")
    parser.add_argument("--data-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--dataset", default="B-dataset")
    parser.add_argument(
        "--split", choices=["random", "cold-drug", "cold-protein", "cold-both"], default="random"
    )
    parser.add_argument("--cold-context", choices=["auxiliary", "strict"], default="auxiliary")
    parser.add_argument("--candidate-scope", choices=["observed", "all"], default="observed")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--all-folds", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    integer_defaults = {
        "hidden-dim": 96,
        "bilinear-heads": 4,
        "bilinear-rank": 32,
        "graph-layers": 2,
        "path-heads": 4,
        "max-paths": 32,
        "diffusion-steps": 100,
        "diffusion-width": 256,
        "diffusion-blocks": 4,
        "ddim-start": 30,
        "ddim-steps": 10,
        "warmup-epochs": 20,
        "diffusion-epochs": 30,
        "epochs": 200,
        "batch-size": 128,
        "eval-batch-size": 256,
        "diffusion-batch-size": 256,
        "patience": 10,
        "max-batches": 0,
        "num-threads": 2,
    }
    float_defaults = {
        "dropout": 0.1,
        "path-degree-penalty": 0.5,
        "condition-dropout": 0.1,
        "residual-weight": 0.1,
        "lr": 1e-3,
        "diffusion-lr": 3e-4,
        "finetune-lr": 1e-4,
        "weight-decay": 1e-4,
        "lambda-diffusion": 0.2,
        "lambda-aux": 0.2,
        "grad-clip": 5.0,
    }
    for name, value in integer_defaults.items():
        parser.add_argument("--" + name, type=int, default=value)
    for name, value in float_defaults.items():
        parser.add_argument("--" + name, type=float, default=value)
    for name in (
        "disable-graph",
        "disable-path",
        "single-predictor",
        "uniform-gate",
        "disable-diffusion",
    ):
        parser.add_argument("--" + name, action="store_true")
    parser.add_argument(
        "--train-condition-encoders", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--enhancement", choices=["entity"], default="entity")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--warmup-checkpoint", type=Path)
    parser.add_argument("--mode", choices=["train", "predict", "evaluate"], default="train")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--pairs-file", type=Path)
    parser.add_argument("--prediction-output", type=Path, default=Path("predictions.csv"))
    return parser
