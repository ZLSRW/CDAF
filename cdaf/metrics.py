"""AUC, AUPR (average precision), Accuracy and F1 at pair and record levels."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_recall_curve,
)


def aggregate_pairs(pairs, labels, probabilities):
    frame = pd.DataFrame(pairs, columns=["drug", "protein"])
    frame["label"], frame["probability"] = labels, probabilities
    if (frame.groupby(["drug", "protein"])["label"].nunique() > 1).any():
        raise ValueError("A candidate pair has contradictory evaluation labels")
    return frame.groupby(["drug", "protein"], as_index=False).agg(
        label=("label", "first"), probability=("probability", "mean")
    )


def choose_threshold(labels, probabilities):
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[np.argmax(f1)])


def classification_metrics(labels, probabilities, threshold):
    predicted = probabilities >= threshold
    both_classes = len(np.unique(labels)) == 2
    return {
        "auc": float(roc_auc_score(labels, probabilities)) if both_classes else None,
        "aupr": float(average_precision_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "threshold": float(threshold),
        "n": len(labels),
        "positives": int(np.sum(labels)),
    }


def evaluate_metrics(pairs, labels, probabilities, threshold):
    grouped = aggregate_pairs(pairs, labels, probabilities)
    return {
        "pairs": classification_metrics(
            grouped.label.to_numpy(), grouped.probability.to_numpy(), threshold
        ),
        "records": classification_metrics(labels, probabilities, threshold),
    }
