"""Load original records, build reproducible grouped splits and disease paths."""

from collections import Counter
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def pair_codes(pairs, n_proteins):
    return pairs[:, 0] * n_proteins + pairs[:, 1]


def read_features(path, expected_dim):
    frame = pd.read_csv(path, header=None)
    values = frame.iloc[:, 1:].to_numpy(dtype=np.float32)
    if values.shape[1] != expected_dim or not np.isfinite(values).all():
        raise ValueError(f"Invalid feature dimensions or nonfinite values in {path}")
    return torch.from_numpy(values)


@dataclass
class Dataset:
    name: str
    drug: torch.Tensor
    protein: torch.Tensor
    disease: torch.Tensor
    dt: np.ndarray
    ds: np.ndarray
    ps: np.ndarray
    drug_ids: list
    protein_ids: list
    hashes: dict

    @property
    def n_drugs(self):
        return len(self.drug)

    @property
    def n_proteins(self):
        return len(self.protein)


def load_dataset(root, name):
    folder = Path(root) / name
    drug = read_features(folder / "Drug_mol2vec.csv", 300)
    protein = read_features(folder / "Protein_ESM.csv", 320)
    disease = read_features(folder / "DiseaseFeature.csv", 64)
    dt = pd.read_csv(folder / "DrugProteinAssociationNumber.csv")[["drug", "protein"]].to_numpy(
        np.int64
    )
    ds = pd.read_csv(folder / "DrugDiseaseAssociationNumber.csv")[["drug", "disease"]].to_numpy(
        np.int64
    )
    ps = pd.read_csv(folder / "ProteinDiseaseAssociationNumber.csv")[
        ["protein", "disease"]
    ].to_numpy(np.int64)
    for edges, limits in [
        (dt, (len(drug), len(protein))),
        (ds, (len(drug), len(disease))),
        (ps, (len(protein), len(disease))),
    ]:
        if len(edges) == 0 or np.any(edges < 0) or np.any(edges >= np.array(limits)):
            raise ValueError(f"Out-of-range or empty association table in {folder}")
    # Node-table order, not the raw information-table row count, defines F indices.
    if (folder / "AllNode.csv").exists():
        nodes = pd.read_csv(folder / "AllNode.csv", header=None)
        if nodes.shape[1] == 1:  # C: no header, one identifier column.
            drug_ids = nodes.iloc[: len(drug), 0].astype(str).tolist()
        else:  # F: header and an explicit index column.
            drug_ids = (
                pd.read_csv(folder / "AllNode.csv")["id"].iloc[: len(drug)].astype(str).tolist()
            )
    else:  # B information table is already in the feature/node order.
        drug_ids = pd.read_csv(folder / "DrugInformation.csv")["id"].astype(str).tolist()
    protein_ids = pd.read_csv(folder / "ProteinInformation.csv")["id"].astype(str).tolist()
    if len(drug_ids) != len(drug) or len(protein_ids) != len(protein):
        raise ValueError("Identifier and feature dimensions disagree")
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.glob("*.csv"))
    }
    return Dataset(name, drug, protein, disease, dt, ds, ps, drug_ids, protein_ids, hashes)


@dataclass
class Split:
    pairs: dict
    labels: dict
    source_rows: dict
    train_edges: np.ndarray
    train_drugs: np.ndarray
    train_proteins: np.ndarray
    ds: np.ndarray
    ps: np.ndarray
    ignored_positive_rows: np.ndarray

    def save(self, directory):
        for key in self.pairs:
            frame = pd.DataFrame(self.pairs[key], columns=["drug", "protein"])
            frame["label"] = self.labels[key].astype(int)
            frame["source_row"] = self.source_rows[key]
            frame.to_csv(Path(directory) / f"split_{key}.csv", index=False)
        np.savez(
            Path(directory) / "context.npz",
            train_edges=self.train_edges,
            train_drugs=self.train_drugs,
            train_proteins=self.train_proteins,
            ds=self.ds,
            ps=self.ps,
            ignored_positive_rows=self.ignored_positive_rows,
        )


def _partitions(values, folds, fold, val_ratio, rng):
    values = rng.permutation(values)
    if len(values) < folds + 2:
        raise ValueError("Too few groups/entities for the requested split")
    pieces = np.array_split(values, folds)
    test = pieces[fold]
    pool = np.concatenate([v for i, v in enumerate(pieces) if i != fold])
    # Validation is local to the outer training pool.
    pool = rng.permutation(pool)
    n_val = max(1, int(round(len(pool) * val_ratio)))
    return {"train": pool[n_val:], "val": pool[:n_val], "test": test}


def make_split(data, args, fold):
    # Same outer permutation for all folds, independent of model RNG.
    positive_codes = pair_codes(data.dt, data.n_proteins)
    unique_positive = np.unique(positive_codes)  # Group keys only; all rows kept below.
    drugs = (
        np.unique(data.dt[:, 0]) if args.candidate_scope == "observed" else np.arange(data.n_drugs)
    )
    proteins = (
        np.unique(data.dt[:, 1])
        if args.candidate_scope == "observed"
        else np.arange(data.n_proteins)
    )
    candidates = (drugs[:, None] * data.n_proteins + proteins[None, :]).reshape(-1)
    unlabeled = candidates[~np.isin(candidates, unique_positive)]
    masks, negative_pools = {}, {}
    train_drugs, train_proteins = drugs, proteins
    if args.split == "random":
        groups = _partitions(
            unique_positive, args.folds, fold, args.val_ratio, np.random.default_rng(args.seed)
        )
        unlab_groups = _partitions(
            unlabeled, args.folds, fold, args.val_ratio, np.random.default_rng(args.seed + 1)
        )
        for key in groups:
            masks[key] = np.isin(positive_codes, groups[key])
            negative_pools[key] = unlab_groups[key]
        train_drugs = np.unique(data.dt[masks["train"], 0])
        train_proteins = np.unique(data.dt[masks["train"], 1])
    else:
        drug_parts = (
            _partitions(
                drugs, args.folds, fold, args.val_ratio, np.random.default_rng(args.seed + 2)
            )
            if args.split in ("cold-drug", "cold-both")
            else None
        )
        protein_parts = (
            _partitions(
                proteins, args.folds, fold, args.val_ratio, np.random.default_rng(args.seed + 3)
            )
            if args.split in ("cold-protein", "cold-both")
            else None
        )
        if drug_parts is not None:
            train_drugs = drug_parts["train"]
        if protein_parts is not None:
            train_proteins = protein_parts["train"]
        for key in ["train", "val", "test"]:
            pos_mask = np.ones(len(data.dt), dtype=bool)
            neg_mask = np.ones(len(unlabeled), dtype=bool)
            if drug_parts is not None:
                pos_mask &= np.isin(data.dt[:, 0], drug_parts[key])
                neg_mask &= np.isin(unlabeled // data.n_proteins, drug_parts[key])
            if protein_parts is not None:
                pos_mask &= np.isin(data.dt[:, 1], protein_parts[key])
                neg_mask &= np.isin(unlabeled % data.n_proteins, protein_parts[key])
            masks[key], negative_pools[key] = pos_mask, unlabeled[neg_mask]
    pairs, labels, rows = {}, {}, {}
    for i, key in enumerate(["train", "val", "test"]):
        positives = data.dt[masks[key]]
        if not len(positives):
            raise ValueError(
                f"{key} contains no positive rows; choose another fold or split configuration"
            )
        count = max(1, int(np.ceil(len(positives) * args.negative_ratio)))
        if len(negative_pools[key]) < count:
            raise ValueError(
                f"Not enough distinct unlabeled pairs in {key} for negative-ratio={args.negative_ratio}"
            )
        sampled = np.random.default_rng(args.seed + fold * 101 + i).choice(
            negative_pools[key], count, replace=False
        )
        negatives = np.column_stack((sampled // data.n_proteins, sampled % data.n_proteins))
        pairs[key] = np.concatenate((positives, negatives))
        labels[key] = np.concatenate((np.ones(len(positives)), np.zeros(count))).astype(np.float32)
        rows[key] = np.concatenate((np.flatnonzero(masks[key]), np.full(count, -1)))
    ds, ps = data.ds.copy(), data.ps.copy()
    if args.cold_context == "strict" and args.split != "random":
        if args.split in ("cold-drug", "cold-both"):
            ds = ds[np.isin(ds[:, 0], train_drugs)]
        if args.split in ("cold-protein", "cold-both"):
            ps = ps[np.isin(ps[:, 0], train_proteins)]
    ignored = np.flatnonzero(~(masks["train"] | masks["val"] | masks["test"]))
    return Split(
        pairs,
        labels,
        rows,
        data.dt[masks["train"]].copy(),
        train_drugs,
        train_proteins,
        ds,
        ps,
        ignored,
    )


class DiseasePaths:
    """Represent duplicate path instances by exact integer multiplicities."""

    def __init__(self, ds, ps, n_drugs, n_proteins, n_diseases, max_paths, seed):
        self.drug = [Counter() for _ in range(n_drugs)]
        self.protein = [Counter() for _ in range(n_proteins)]
        for v, s in ds:
            self.drug[v][s] += 1
        for v, s in ps:
            self.protein[v][s] += 1
        self.degree = np.bincount(np.concatenate((ds[:, 1], ps[:, 1])), minlength=n_diseases)
        self.max_paths, self.seed, self.n_proteins = max_paths, seed, n_proteins

    def batch(self, pairs, device):
        pairs = np.asarray(pairs, dtype=np.int64)
        ids = np.zeros((len(pairs), self.max_paths), dtype=np.int64)
        valid = np.zeros_like(ids, dtype=bool)
        counts = np.zeros(len(pairs), dtype=np.int64)
        for row, (d, p) in enumerate(pairs):
            left, right = self.drug[d], self.protein[p]
            shared = sorted(left.keys() & right.keys())
            if not shared:
                continue
            multiplicities = np.array([left[s] * right[s] for s in shared], dtype=np.int64)
            total = int(multiplicities.sum())
            counts[row] = total
            # Sampling instance indices avoids materializing a huge Cartesian product.
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(d), int(p)]))
            positions = (
                np.arange(total)
                if total <= self.max_paths
                else rng.choice(total, self.max_paths, replace=False)
            )
            chosen = np.asarray(shared)[
                np.searchsorted(np.cumsum(multiplicities), positions, side="right")
            ]
            ids[row, : len(chosen)] = chosen
            valid[row, : len(chosen)] = True
        return (
            torch.as_tensor(ids, device=device),
            torch.as_tensor(valid, device=device),
            torch.as_tensor(counts, device=device),
        )


def mask_target_edges(edges, pairs, labels, n_proteins):
    """Remove every record for each supervised positive, before either direction is built."""
    targets = pairs[labels > 0.5]
    if not len(targets):
        return edges
    codes = edges[:, 0] * n_proteins + edges[:, 1]
    target_codes = targets[:, 0] * n_proteins + targets[:, 1]
    return edges[~torch.isin(codes, target_codes)]
