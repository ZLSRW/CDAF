"""Reproducible orchestration of the fixed manuscript configuration."""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import uuid

import numpy as np
import pandas as pd
import torch

from .config import build_parser
from .data import load_dataset, make_split, DiseasePaths
from .metrics import evaluate_metrics
from .model import CDAF
from .training import (
    run_fold,
    setup,
    tensor_inputs,
    score_pairs,
    prediction_frame,
    predict_checkpoint,
    write_json,
    validate_args,
)

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("B-dataset", "C-dataset", "F-dataset")
METRICS = ("auc", "aupr", "accuracy", "f1")


def configuration(name, options):
    parser = build_parser()
    args = parser.parse_args([])
    values = json.loads((ROOT / "configs" / f"{name}.json").read_text())
    for key, value in values.items():
        setattr(args, key, value)
    for key in (
        "epochs",
        "warmup_epochs",
        "diffusion_epochs",
        "hidden_dim",
        "finetune_lr",
        "lambda_diffusion",
        "lambda_aux",
    ):
        value = getattr(options, key, None)
        if value is not None:
            setattr(args, key, value)
    args.dataset = name
    args.device, args.num_threads = options.device, options.num_threads
    args.data_root, args.output_dir = options.data_root.resolve(), options.output_dir.resolve()
    args.validation_only, args.all_folds = True, False
    args.warmup_checkpoint = None
    if options.smoke:
        args.warmup_epochs = args.diffusion_epochs = args.epochs = args.max_batches = 1
    validate_args(args, parser)
    return args


def verify_data(root, names):
    manifest = json.loads((ROOT / "reference/data_sha256.json").read_text())
    checked = 0
    for rel, expected in manifest.items():
        path = Path(rel)
        if path.parts[1] not in names:
            continue
        source = root.joinpath(*path.parts[1:])
        if not source.is_file():
            raise FileNotFoundError(f"Required data file is missing: {source}")
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Data checksum mismatch: {source}")
        checked += 1
    print(f"Verified {checked} original CSV files.", flush=True)


def verify_split(data, args, fold):
    actual = make_split(data, args, fold)
    folder = ROOT / "reference" / data.name / f"fold_{fold}"
    for key in ("train", "val", "test"):
        expected = pd.read_csv(folder / f"split_{key}.csv")
        np.testing.assert_array_equal(actual.pairs[key], expected[["drug", "protein"]].to_numpy())
        np.testing.assert_array_equal(actual.labels[key], expected.label.to_numpy())
        np.testing.assert_array_equal(actual.source_rows[key], expected.source_row.to_numpy())
    with np.load(folder / "context.npz") as context:
        for key in context.files:
            np.testing.assert_array_equal(getattr(actual, key), context[key])
    return actual


def summarize(rows, directory):
    write_json(directory / "results.json", rows)
    summary = []
    for name in sorted({r["dataset"] for r in rows}):
        selected = [r for r in rows if r["dataset"] == name]
        row = dict(
            dataset=name, folds=len(selected), evaluation_split=selected[0]["evaluation_split"]
        )
        for key in METRICS:
            values = [r["pairs"][key] for r in selected]
            row[key + "_mean"] = float(np.mean(values))
            row[key + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        summary.append(row)
    pd.DataFrame(summary).to_csv(directory / "summary.csv", index=False)


def checkpoint_path(weights_root, name, fold):
    """Accept both this release's training layout and a flat checkpoint folder."""
    folder = weights_root / name / f"fold_{fold}"
    for candidate in (folder / "final" / "best.pt", folder / "best.pt"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No trained checkpoint in {folder}. Run train mode first and pass its run directory to --weights-root."
    )


def evaluate(data, args, fold, checkpoint_path, directory, split_name):
    device = setup(args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["dataset"] != data.name or checkpoint["fold"] != fold:
        raise ValueError("Checkpoint dataset/fold does not match the request")
    if checkpoint["hashes"] != data.hashes:
        raise ValueError("Checkpoint input hashes do not match the original CSV files")
    config = argparse.Namespace(**checkpoint["config"])
    split = verify_split(data, config, fold)
    for key, values in (("train_edges", split.train_edges), ("ds", split.ds), ("ps", split.ps)):
        np.testing.assert_array_equal(checkpoint[key].numpy(), values)
    model = CDAF(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    tensors = tensor_inputs(data, split.ds, split.ps, device)
    paths = DiseasePaths(
        split.ds,
        split.ps,
        data.n_drugs,
        data.n_proteins,
        len(data.disease),
        config.max_paths,
        config.seed + fold,
    )
    scored = score_pairs(
        model,
        tensors,
        split.pairs[split_name],
        checkpoint["train_edges"].to(device),
        paths,
        torch.as_tensor(paths.degree, device=device, dtype=torch.float32),
        config.eval_batch_size,
        checkpoint["use_diffusion"],
        config.seed + 10000 + fold,
    )
    frame = prediction_frame(
        data, split.pairs[split_name], scored, checkpoint["threshold"], split.labels[split_name]
    )
    frame.to_csv(directory / f"{split_name}_predictions.csv", index=False)
    metrics = evaluate_metrics(
        split.pairs[split_name],
        split.labels[split_name],
        scored["probability"],
        checkpoint["threshold"],
    )
    metrics.update(dataset=data.name, fold=fold, evaluation_split=split_name)
    if split_name == "val":
        expected = pd.read_csv(
            ROOT / "reference" / data.name / f"fold_{fold}/validation_predictions.csv",
            dtype={"score": np.float32},
        )
        np.testing.assert_array_equal(
            frame[["drug", "protein", "label"]], expected[["drug", "protein", "label"]]
        )
        metrics["max_score_error_vs_paper"] = float(
            np.max(np.abs(frame.score.to_numpy() - expected.score.to_numpy()))
        )
    write_json(directory / "metrics.json", metrics)
    print(data.name, f"fold={fold}", metrics["pairs"], flush=True)
    return metrics


def run(options):
    names = DATASETS if options.dataset == "all" else (options.dataset,)
    if options.mode == "predict":
        checkpoint = torch.load(options.checkpoint, map_location="cpu", weights_only=True)
        name = checkpoint["dataset"]
        verify_data(options.data_root, (name,))
        args = configuration(name, options)
        args.seed = checkpoint["config"]["seed"]
        args.eval_batch_size = checkpoint["config"]["eval_batch_size"]
        args.checkpoint = options.checkpoint
        args.pairs_file = options.pairs_file
        args.prediction_output = options.prediction_output
        predict_checkpoint(args)
        return
    verify_data(options.data_root, names)
    if options.mode == "verify-data":
        for name in names:
            data = load_dataset(options.data_root, name)
            args = configuration(name, options)
            for fold in options.folds:
                verify_split(data, args, fold)
        print("All requested fold records and graph contexts match the paper references.")
        return
    if options.mode == "evaluate":
        for name in names:
            for fold in options.folds:
                checkpoint_path(options.weights_root, name, fold)
    run_id = options.run_name or (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + options.mode + "_" + uuid.uuid4().hex[:6]
    )
    directory = options.output_dir / run_id
    directory.mkdir(parents=True, exist_ok=False)
    write_json(
        directory / "environment.json",
        dict(
            python=platform.python_version(),
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            numpy=np.__version__,
            pandas=pd.__version__,
            device=options.device,
            gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            started_utc=datetime.now(timezone.utc).isoformat(),
            smoke=options.smoke,
        ),
    )
    rows = []
    for name in names:
        data = load_dataset(options.data_root, name)
        args = configuration(name, options)
        for fold in options.folds:
            args.fold = fold
            verify_split(data, args, fold)
            folder = directory / name / f"fold_{fold}"
            folder.mkdir(parents=True)
            if options.mode == "evaluate":
                path = checkpoint_path(options.weights_root, name, fold)
                metrics = evaluate(data, args, fold, path, folder, options.eval_split)
            else:
                warmup_args = copy.deepcopy(args)
                warmup_args.disable_diffusion, warmup_args.epochs = True, 0
                warmup_args.warmup_checkpoint = None
                warmup_folder = folder / "warmup"
                warmup_folder.mkdir()
                run_fold(warmup_args, data, fold, warmup_folder, setup(warmup_args))
                warmup_path = warmup_folder / "warmup_best.pt"
                final_args = copy.deepcopy(args)
                final_args.warmup_checkpoint = warmup_path
                final_folder = folder / "final"
                final_folder.mkdir()
                # The paper runs restarted with seed=42 before loading the warmup.
                metrics = run_fold(final_args, data, fold, final_folder, setup(final_args))
            rows.append(metrics)
            summarize(rows, directory)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"Results: {directory}", flush=True)
