"""Training stages, evaluation, experiment persistence and checkpoint inference."""

import argparse
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import pandas as pd
import torch
from .data import DiseasePaths, load_dataset, make_split, mask_target_edges
from .metrics import aggregate_pairs, choose_threshold, evaluate_metrics
from .model import CDAF, classification_loss


def validate_args(args, parser):
    positive = [
        "hidden_dim",
        "bilinear_heads",
        "bilinear_rank",
        "graph_layers",
        "path_heads",
        "max_paths",
        "diffusion_steps",
        "diffusion_width",
        "diffusion_blocks",
        "ddim_start",
        "ddim_steps",
        "batch_size",
        "eval_batch_size",
        "diffusion_batch_size",
        "num_threads",
        "patience",
    ]
    for name in positive:
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.folds < 2 or not 0 <= args.fold < args.folds:
        parser.error("Require folds >= 2 and 0 <= fold < folds")
    if not 0 < args.val_ratio < 0.5 or args.negative_ratio <= 0 or args.seed < 0:
        parser.error("Require 0 < val-ratio < 0.5, negative-ratio > 0, seed >= 0")
    if args.hidden_dim % args.path_heads:
        parser.error("hidden-dim must be divisible by path-heads")
    if not 1 <= args.ddim_steps <= args.ddim_start <= args.diffusion_steps:
        parser.error("Require 1 <= ddim-steps <= ddim-start <= diffusion-steps")
    if not 0 < args.residual_weight < 1:
        parser.error("residual-weight must be in (0, 1)")
    if not 0 <= args.dropout < 1 or not 0 <= args.condition_dropout < 1:
        parser.error("Dropout probabilities must be in [0, 1)")
    if min(args.lr, args.diffusion_lr, args.finetune_lr, args.grad_clip) <= 0:
        parser.error("Learning rates and grad-clip must be positive")
    if (
        min(
            args.weight_decay,
            args.lambda_diffusion,
            args.lambda_aux,
            args.path_degree_penalty,
            args.max_batches,
            args.epochs,
        )
        < 0
    ):
        parser.error("Loss weights, penalties, max-batches and epochs must be nonnegative")
    if args.mode == "train":
        if args.warmup_epochs < 1:
            parser.error("warmup-epochs must be at least 1")
        if not args.disable_diffusion and (args.diffusion_epochs < 1 or args.epochs < 1):
            parser.error("Diffusion requires diffusion-epochs >= 1 and epochs >= 1")
    elif args.mode == "evaluate":
        if args.checkpoint is None:
            parser.error("evaluate requires --checkpoint")
    elif args.checkpoint is None or args.pairs_file is None:
        parser.error("predict requires --checkpoint and --pairs-file")
    if getattr(args, "single_predictor", False) and (
        args.disable_graph or args.disable_path or args.uniform_gate
    ):
        parser.error(
            "--single-predictor removes all channels; do not combine with channel/gate ablations"
        )


def serializable_args(args):
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}


def write_json(path, obj):
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def setup(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.num_threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and (not torch.cuda.is_available()):
        raise ValueError("CUDA requested but unavailable in this Python environment")
    return device


def tensor_inputs(data, ds, ps, device):
    return {
        "drug": data.drug.to(device),
        "protein": data.protein.to(device),
        "disease": data.disease.to(device),
        "ds": torch.as_tensor(ds, dtype=torch.long, device=device),
        "ps": torch.as_tensor(ps, dtype=torch.long, device=device),
    }


def sampled_diffusion_loss(model, encoded, train_indices, size, train_pairs=None):
    terms = []
    for kind, name in enumerate(["drug", "protein"]):
        candidates = train_indices[kind]
        count = min(size, len(candidates))
        indices = candidates[torch.randperm(len(candidates), device=candidates.device)[:count]]
        terms.append(
            model.diffusion.noise_loss(
                encoded[name][indices],
                encoded[name + "_context"][indices],
                encoded[name + "_available"][indices],
                kind,
            )
        )
    return torch.stack(terms).mean()


@torch.no_grad()
def score_pairs(
    model, tensors, pairs, edges, paths, disease_degree, batch_size, use_diffusion, noise_seed
):
    model.eval()
    device = tensors["drug"].device
    encoded = model.encode(tensors)
    enhanced = model.enhance(encoded, use_diffusion, noise_seed)
    graph = model.prepare_graph(enhanced, edges)
    predictions, gates, available, channels, heads, top_diseases, top_weights = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        tensor_pairs = torch.as_tensor(batch, dtype=torch.long, device=device)
        path_data = None if model.single_predictor else paths.batch(batch, device)
        output = model.score(
            tensor_pairs,
            encoded,
            enhanced,
            graph,
            path_data,
            disease_degree,
            use_diffusion,
            noise_seed,
        )
        predictions.append(output["probability"].cpu().numpy())
        if model.single_predictor:
            continue
        gates.append(output["gate"].cpu().numpy())
        available.append(output["available"].cpu().numpy())
        channels.append(output["channel_logits"].sigmoid().cpu().numpy())
        heads.append(output["head_weights"].cpu().numpy())
        top_slot = output["path_weights"].argmax(-1)
        row = torch.arange(len(batch), device=device)
        top_disease = output["path_ids"][row, top_slot]
        top_disease = torch.where(output["available"][:, 2], top_disease, -1)
        top_diseases.append(top_disease.cpu().numpy())
        top_weights.append(output["path_weights"][row, top_slot].cpu().numpy())
    extra = {}
    if model.single_predictor:
        return dict(probability=np.concatenate(predictions), **extra)
    return dict(
        probability=np.concatenate(predictions),
        gate=np.concatenate(gates),
        available=np.concatenate(available),
        channels=np.concatenate(channels),
        head_weights=np.concatenate(heads),
        top_disease=np.concatenate(top_diseases),
        top_path_weight=np.concatenate(top_weights),
        **extra,
    )


def prediction_frame(data, pairs, scored, threshold, labels=None):
    frame = pd.DataFrame(pairs, columns=["drug", "protein"])
    frame["drug_id"] = [data.drug_ids[d] for d in pairs[:, 0]]
    frame["protein_id"] = [data.protein_ids[p] for p in pairs[:, 1]]
    if labels is not None:
        frame["label"] = labels.astype(int)
    frame["score"] = scored["probability"]
    frame["prediction"] = (frame.score >= threshold).astype(int)
    if "gate" not in scored:
        return frame
    for i, name in enumerate(["bilinear", "graph", "path"]):
        frame[f"weight_{name}"] = scored["gate"][:, i]
        frame[f"score_{name}"] = scored["channels"][:, i]
        frame[f"available_{name}"] = scored["available"][:, i]
    frame["top_disease_index"] = scored["top_disease"]
    frame["top_path_weight"] = scored["top_path_weight"]
    return frame


def _checkpoint(model, args, data, split, stage, epoch, use_diffusion, threshold, fold):
    result = {
        "model_state": model.state_dict(),
        "config": serializable_args(args),
        "dataset": data.name,
        "hashes": data.hashes,
        "stage": stage,
        "epoch": epoch,
        "fold": fold,
        "use_diffusion": use_diffusion,
        "threshold": float(threshold),
        "train_edges": torch.from_numpy(split.train_edges),
        "ds": torch.from_numpy(split.ds),
        "ps": torch.from_numpy(split.ps),
    }
    return result


def run_fold(args, data, fold, directory, device):
    started = time.time()
    split = make_split(data, args, fold)
    split.save(directory)
    write_json(
        directory / "config.json", serializable_args(args) | {"dataset": data.name, "fold": fold}
    )
    write_json(
        directory / "data_manifest.json",
        {
            "sha256": data.hashes,
            "n_drugs": data.n_drugs,
            "n_proteins": data.n_proteins,
            "n_diseases": len(data.disease),
            "raw_dti_rows": len(data.dt),
            "ignored_cross_partition_rows": len(split.ignored_positive_rows),
            "split_rows": {k: len(v) for k, v in split.pairs.items()},
            "negative_semantics": "sampled unlabeled pairs; not experimentally verified negatives",
            "selection_metric": "pair-level AUPR on validation set",
            "aupr_definition": "sklearn.metrics.average_precision_score (not trapezoidal PR-AUC)",
        },
    )
    tensors = tensor_inputs(data, split.ds, split.ps, device)
    context_edges = torch.as_tensor(split.train_edges, device=device)
    paths = DiseasePaths(
        split.ds,
        split.ps,
        data.n_drugs,
        data.n_proteins,
        len(data.disease),
        args.max_paths,
        args.seed + fold,
    )
    disease_degree = torch.as_tensor(paths.degree, device=device, dtype=torch.float32)
    model = CDAF(args).to(device)
    train_pairs = torch.as_tensor(split.pairs["train"], device=device)
    train_labels = torch.as_tensor(split.labels["train"], device=device)
    training_paths = None if model.single_predictor else paths.batch(split.pairs["train"], device)
    train_indices = [
        torch.as_tensor(v, device=device) for v in (split.train_drugs, split.train_proteins)
    ]
    noise_seed = args.seed + 10000 + fold
    history = []
    frozen_encoding = None
    n_batches = math.ceil(len(train_pairs) / args.batch_size)
    print(
        f"[{data.name} fold={fold}] device={device}, parameters={sum((p.numel() for p in model.parameters())):,}, train/val/test records={[len(split.pairs[k]) for k in ('train', 'val', 'test')]}",
        flush=True,
    )

    def record(entry):
        history.append(entry)
        with (directory / "history.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")

    def classification_stage(stage, epochs, lr, use_diffusion):
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=args.weight_decay,
        )
        best, stale = (-1.0, 0)
        stage_path = directory / f"{stage}_best.pt"
        for epoch in range(1, epochs + 1):
            epoch_started = time.time()
            model.train()
            order = torch.randperm(len(train_pairs), device=device)
            losses = []
            for batch_index, start in enumerate(range(0, len(train_pairs), args.batch_size)):
                if args.max_batches and batch_index >= args.max_batches:
                    break
                idx = order[start : start + args.batch_size]
                pairs, labels = (train_pairs[idx], train_labels[idx])
                graph_edges = mask_target_edges(context_edges, pairs, labels, data.n_proteins)
                path_data = (
                    None
                    if training_paths is None
                    else tuple((value[idx] for value in training_paths))
                )
                optimizer.zero_grad(set_to_none=True)
                batch_encoding = model.refresh_conditions(frozen_encoding, tensors)
                output = model(
                    tensors,
                    pairs,
                    graph_edges,
                    path_data,
                    disease_degree,
                    use_diffusion=use_diffusion,
                    encoded=batch_encoding,
                )
                loss = classification_loss(output, labels, args.lambda_aux)
                if use_diffusion:
                    auxiliary = sampled_diffusion_loss(
                        model, batch_encoding, train_indices, args.diffusion_batch_size, train_pairs
                    )
                    loss = loss + args.lambda_diffusion * auxiliary
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite {stage} loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip, error_if_nonfinite=True
                )
                optimizer.step()
                losses.append(float(loss.detach()))
            validation = score_pairs(
                model,
                tensors,
                split.pairs["val"],
                context_edges,
                paths,
                disease_degree,
                args.eval_batch_size,
                use_diffusion,
                noise_seed,
            )
            grouped = aggregate_pairs(
                split.pairs["val"], split.labels["val"], validation["probability"]
            )
            threshold = choose_threshold(grouped.label.to_numpy(), grouped.probability.to_numpy())
            metrics = evaluate_metrics(
                split.pairs["val"], split.labels["val"], validation["probability"], threshold
            )
            aupr = metrics["pairs"]["aupr"]
            entry = dict(
                stage=stage,
                epoch=epoch,
                loss=float(np.mean(losses)),
                val_aupr=aupr,
                val_auc=metrics["pairs"]["auc"],
                val_accuracy=metrics["pairs"]["accuracy"],
                val_f1=metrics["pairs"]["f1"],
                seconds=time.time() - epoch_started,
                batches=len(losses),
                full_epoch_batches=n_batches,
                mean_gate=validation["gate"].mean(0).tolist() if "gate" in validation else None,
            )
            record(entry)
            print(
                f"  {stage} {epoch}/{epochs} loss={entry['loss']:.4f} val_AUC={entry['val_auc']:.4f} val_AUPR={aupr:.4f} val_Accuracy={entry['val_accuracy']:.4f} val_F1={entry['val_f1']:.4f} time={entry['seconds']:.1f}s",
                flush=True,
            )
            if aupr > best:
                best, stale = (aupr, 0)
                torch.save(
                    _checkpoint(
                        model, args, data, split, stage, epoch, use_diffusion, threshold, fold
                    ),
                    stage_path,
                )
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"  Early stopping {stage}", flush=True)
                    break
        checkpoint = torch.load(stage_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        return checkpoint

    if getattr(args, "warmup_checkpoint", None):
        selected = torch.load(args.warmup_checkpoint, map_location=device, weights_only=True)
        if (
            selected["dataset"] != data.name
            or selected["hashes"] != data.hashes
            or selected["fold"] != fold
            or (selected["stage"] != "warmup")
            or selected["use_diffusion"]
        ):
            raise ValueError("Warmup checkpoint does not match this dataset/fold/stage")
        for key, expected in [
            ("train_edges", split.train_edges),
            ("ds", split.ds),
            ("ps", split.ps),
        ]:
            if not np.array_equal(selected[key].cpu().numpy(), expected):
                raise ValueError("Warmup checkpoint context differs from this split")
        for key in [
            "seed",
            "split",
            "folds",
            "val_ratio",
            "negative_ratio",
            "candidate_scope",
            "cold_context",
            "single_predictor",
        ]:
            if selected["config"].get(key, False) != getattr(args, key):
                raise ValueError(f"Warmup checkpoint configuration differs: {key}")
        model.load_state_dict(selected["model_state"])
        selected = _checkpoint(
            model,
            args,
            data,
            split,
            "warmup",
            selected["epoch"],
            False,
            selected["threshold"],
            fold,
        )
        torch.save(selected, directory / "warmup_best.pt")
        record(
            dict(
                stage="warmup_reused",
                checkpoint=str(args.warmup_checkpoint),
                epoch=selected["epoch"],
            )
        )
    else:
        selected = classification_stage("warmup", args.warmup_epochs, args.lr, False)
    model.freeze_encoders()
    with torch.no_grad():
        frozen_encoding = model.encode(tensors)
    if not args.disable_diffusion:
        with torch.no_grad():
            model.diffusion.fit_stats(
                frozen_encoding["drug"], frozen_encoding["protein"], *train_indices
            )
    if not args.disable_diffusion:
        diffusion_parameters = list(model.diffusion.parameters())
        if model.train_condition_encoders:
            diffusion_parameters += model.condition_parameters()
        optimizer = torch.optim.AdamW(
            diffusion_parameters, lr=args.diffusion_lr, weight_decay=args.weight_decay
        )
        diffusion_batches = math.ceil(max(map(len, train_indices)) / args.diffusion_batch_size)
        for epoch in range(1, args.diffusion_epochs + 1):
            model.train()
            losses = []
            for step in range(diffusion_batches):
                if args.max_batches and step >= args.max_batches:
                    break
                optimizer.zero_grad(set_to_none=True)
                batch_encoding = model.refresh_conditions(frozen_encoding, tensors)
                loss = sampled_diffusion_loss(
                    model, batch_encoding, train_indices, args.diffusion_batch_size, train_pairs
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite diffusion loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    diffusion_parameters, args.grad_clip, error_if_nonfinite=True
                )
                optimizer.step()
                losses.append(float(loss.detach()))
            record(
                dict(
                    stage="diffusion", epoch=epoch, loss=float(np.mean(losses)), batches=len(losses)
                )
            )
            objective = "epsilon_MSE"
            print(
                f"  diffusion {epoch}/{args.diffusion_epochs} {objective}={np.mean(losses):.4f}",
                flush=True,
            )
        torch.save(
            _checkpoint(
                model,
                args,
                data,
                split,
                "diffusion",
                args.diffusion_epochs,
                True,
                selected["threshold"],
                fold,
            ),
            directory / "diffusion_pretrained.pt",
        )
    if args.epochs:
        selected = classification_stage(
            "finetune", args.epochs, args.finetune_lr, not args.disable_diffusion
        )
    torch.save(selected, directory / "best.pt")
    evaluation_split = "val" if getattr(args, "validation_only", False) else "test"
    prefix = "validation" if evaluation_split == "val" else "test"
    test = score_pairs(
        model,
        tensors,
        split.pairs[evaluation_split],
        context_edges,
        paths,
        disease_degree,
        args.eval_batch_size,
        selected["use_diffusion"],
        noise_seed,
    )
    metrics = evaluate_metrics(
        split.pairs[evaluation_split],
        split.labels[evaluation_split],
        test["probability"],
        selected["threshold"],
    )
    metrics.update(
        evaluation_split=evaluation_split,
        enhancement="entity",
        dataset=data.name,
        fold=fold,
        elapsed_seconds=time.time() - started,
        graph_coverage=float(test["available"][:, 1].mean()) if "available" in test else None,
        path_coverage=float(test["available"][:, 2].mean()) if "available" in test else None,
        mean_gate=test["gate"].mean(0).tolist() if "gate" in test else None,
        predictor="single_pair_mlp" if model.single_predictor else "availability_aware_fusion",
        diffusion_enabled=not args.disable_diffusion,
        encoders_frozen_after_warmup=not model.train_condition_encoders,
        base_encoders_frozen_after_warmup=True,
        condition_encoders_trained=model.train_condition_encoders,
        debug_limited_batches=bool(args.max_batches),
        checkpoint=str(directory / "best.pt"),
    )
    write_json(directory / "metrics.json", metrics)
    prediction_frame(
        data,
        split.pairs[evaluation_split],
        test,
        selected["threshold"],
        split.labels[evaluation_split],
    ).to_csv(directory / f"{prefix}_predictions.csv", index=False)
    if not model.single_predictor:
        np.savez(
            directory / f"{prefix}_explanations.npz",
            bilinear_head_weights=test["head_weights"],
            top_disease_index=test["top_disease"],
            top_path_weight=test["top_path_weight"],
        )
    print(
        f"  {prefix.title()} AUC={metrics['pairs']['auc']:.4f}, AUPR={metrics['pairs']['aupr']:.4f}, Accuracy={metrics['pairs']['accuracy']:.4f}, F1={metrics['pairs']['f1']:.4f}; saved to {directory}",
        flush=True,
    )
    return metrics


def predict_checkpoint(args):
    device = setup(args)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = argparse.Namespace(**checkpoint["config"])
    data = load_dataset(args.data_root, checkpoint["dataset"])
    if data.hashes != checkpoint["hashes"]:
        raise ValueError("Dataset files differ from the checkpoint; use its original data root")
    pairs_frame = pd.read_csv(args.pairs_file)
    raw = pairs_frame[["drug", "protein"]].to_numpy()
    if not np.isfinite(raw.astype(float)).all() or not np.equal(raw, raw.astype(np.int64)).all():
        raise ValueError("Prediction pair indices must be finite integers")
    pairs = raw.astype(np.int64)
    if not len(pairs) or np.any(pairs < 0) or np.any(pairs >= [data.n_drugs, data.n_proteins]):
        raise ValueError("Prediction pairs are empty or out of node index range")
    ds, ps = (checkpoint["ds"].numpy(), checkpoint["ps"].numpy())
    tensors = tensor_inputs(data, ds, ps, device)
    model = CDAF(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    edges = checkpoint["train_edges"].to(device)
    paths = DiseasePaths(
        ds,
        ps,
        data.n_drugs,
        data.n_proteins,
        len(data.disease),
        config.max_paths,
        config.seed + checkpoint["fold"],
    )
    degree = torch.as_tensor(paths.degree, device=device, dtype=torch.float32)
    scored = score_pairs(
        model,
        tensors,
        pairs,
        edges,
        paths,
        degree,
        args.eval_batch_size,
        checkpoint["use_diffusion"],
        config.seed + 10000 + checkpoint["fold"],
    )
    frame = prediction_frame(data, pairs, scored, checkpoint["threshold"])
    args.prediction_output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.prediction_output, index=False)
    print(f"Saved {len(frame)} DTI predictions to {args.prediction_output}", flush=True)
