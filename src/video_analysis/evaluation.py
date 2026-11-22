"""Accuracy metrics against manual labels, for spot-checking results on a sample."""

from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment


def boundary_metrics(predicted_s: list[float], expected_s: list[float], tolerance_s: float) -> dict:
    """Precision, recall and F1 for cut or scene-start times.

    Each expected boundary can be matched by at most one predicted boundary
    within ``tolerance_s``; the optimal one-to-one matching is used so a
    single prediction cannot be credited twice.
    """
    if tolerance_s < 0:
        raise ValueError("tolerance_s cannot be negative")
    matched_errors: list[float] = []
    if predicted_s and expected_s:
        distances = np.abs(np.subtract.outer(predicted_s, expected_s))
        # Out-of-tolerance pairs get a prohibitive cost, so the assignment maximises matches first.
        costs = np.where(distances <= tolerance_s, distances, 1e9)
        rows, columns = linear_sum_assignment(costs)
        matched_errors = [float(distances[row, column]) for row, column in zip(rows, columns, strict=True)
                          if distances[row, column] <= tolerance_s]
    matched = len(matched_errors)
    precision = matched / len(predicted_s) if predicted_s else float(not expected_s)
    recall = matched / len(expected_s) if expected_s else 1.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "matched": matched,
        "predicted": len(predicted_s),
        "expected": len(expected_s),
        "mean_abs_error_s": sum(matched_errors) / matched if matched else None,
    }


def grouping_metrics(true_people: list[str], predicted_groups: list[str]) -> dict:
    """Score face groups against manual person labels for the same faces.

    ``purity`` is the share of faces whose group's most common person is their
    own person; it drops when different people are merged into one group.
    ``inverse_purity`` is the share of faces whose person's most common group
    is their own group; it drops when one person is split across groups.
    """
    if len(true_people) != len(predicted_groups) or not true_people:
        raise ValueError("Provide one manual label per predicted face")

    def majority_share(keys: list[str], values: list[str]) -> float:
        best: dict[str, int] = {}
        for (key, _), count in Counter(zip(keys, values, strict=True)).items():
            best[key] = max(best.get(key, 0), count)
        return sum(best.values()) / len(keys)

    return {
        "purity": majority_share(predicted_groups, true_people),
        "inverse_purity": majority_share(true_people, predicted_groups),
        "faces": len(true_people),
        "people": len(set(true_people)),
        "groups": len(set(predicted_groups)),
    }
