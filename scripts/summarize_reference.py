"""Recompute reference metrics from archived predictions, without rerunning inference."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cdaf.metrics import evaluate_metrics


def main():
    rows = []
    for dataset in ("B-dataset", "C-dataset", "F-dataset"):
        for fold in range(5):
            directory = ROOT / "reference" / dataset / f"fold_{fold}"
            saved = json.loads((directory / "metrics.json").read_text())
            # Training exports float32 scores using their shortest decimal form.
            # Restore the original dtype before aggregation/threshold comparison.
            frame = pd.read_csv(
                directory / "validation_predictions.csv", dtype={"score": np.float32}
            )
            metrics = evaluate_metrics(
                frame[["drug", "protein"]].to_numpy(),
                frame.label.to_numpy(),
                frame.score.to_numpy(),
                saved["pairs"]["threshold"],
            )
            for key in ("auc", "aupr", "accuracy", "f1"):
                np.testing.assert_allclose(
                    metrics["pairs"][key], saved["pairs"][key], atol=1e-12, rtol=0
                )
            rows.append(dict(dataset=dataset, fold=fold, **metrics["pairs"]))
    frame = pd.DataFrame(rows)
    print("Recomputed all 15 archived validation folds; metrics match reference JSON files.")
    print("Dataset | AUC | AUPR | Accuracy | F1")
    for dataset, group in frame.groupby("dataset", sort=True):
        cells = [
            f"{group[k].mean():.4f} +/- {group[k].std(ddof=1):.4f}"
            for k in ("auc", "aupr", "accuracy", "f1")
        ]
        print(dataset + " | " + " | ".join(cells))


if __name__ == "__main__":
    main()
