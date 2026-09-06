"""Метрики: подбор порога, ROC-AUC по вероятностям, реестр результатов."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score, roc_auc_score

from deepfake.metrics import (
    ExperimentRegistry,
    ExperimentResult,
    classification_metrics,
    f1_over_thresholds,
)


def test_tuned_threshold_beats_half(labelled_probabilities):
    """Главный смысл подбора: при дисбалансе порог 0.5 систематически хуже."""
    labels, probs = labelled_probabilities

    tuned_f1, threshold = f1_over_thresholds(labels, probs)
    f1_at_half = f1_score(labels, (probs >= 0.5).astype(int), zero_division=0)

    assert tuned_f1 > f1_at_half
    assert 0.0 < threshold < 1.0


def test_tuned_f1_is_the_maximum(labelled_probabilities):
    """Найденный F1 не должен уступать ни одному порогу из плотной сетки."""
    labels, probs = labelled_probabilities
    tuned_f1, _ = f1_over_thresholds(labels, probs)

    grid_best = max(
        f1_score(labels, (probs >= t).astype(int), zero_division=0)
        for t in np.linspace(0.01, 0.99, 199)
    )

    assert tuned_f1 >= grid_best - 1e-9


def test_perfect_ranking_gives_perfect_f1():
    labels = np.array([0, 0, 0, 1, 1])
    probs = np.array([0.01, 0.02, 0.03, 0.9, 0.95])

    tuned_f1, _ = f1_over_thresholds(labels, probs)

    assert tuned_f1 == pytest.approx(1.0)


def test_roc_auc_uses_probabilities_not_labels(labelled_probabilities):
    """AUC от жёстких меток — это не ROC-AUC, а балансированная точность."""
    labels, probs = labelled_probabilities

    metrics = classification_metrics(labels, probs, threshold=0.5)
    auc_from_labels = roc_auc_score(labels, (probs >= 0.5).astype(int))

    assert metrics["roc_auc"] == pytest.approx(roc_auc_score(labels, probs))
    assert metrics["roc_auc"] > auc_from_labels + 0.1


def test_roc_auc_is_threshold_independent(labelled_probabilities):
    labels, probs = labelled_probabilities

    low = classification_metrics(labels, probs, threshold=0.2)["roc_auc"]
    high = classification_metrics(labels, probs, threshold=0.8)["roc_auc"]

    assert low == pytest.approx(high)


def test_classification_metrics_are_in_range(labelled_probabilities):
    labels, probs = labelled_probabilities
    metrics = classification_metrics(labels, probs, threshold=0.3)

    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert 0.0 <= metrics[key] <= 1.0, key


def test_registry_sorts_by_f1():
    registry = ExperimentRegistry()
    registry.add(_result("worse", f1=0.70))
    registry.add(_result("better", f1=0.91))

    assert list(registry.to_frame().index) == ["better", "worse"]


def test_registry_roundtrip(tmp_path):
    registry = ExperimentRegistry()
    registry.add(_result("model", f1=0.8))

    path = tmp_path / "results.csv"
    registry.save(path)
    restored = ExperimentRegistry.load(path)

    assert restored.results["model"].f1 == pytest.approx(0.8)
    assert restored.results["model"].note == "заметка"


def test_registry_load_missing_file_is_empty(tmp_path):
    registry = ExperimentRegistry.load(tmp_path / "nothing.csv")
    assert registry.to_frame().empty


def _result(name: str, f1: float) -> ExperimentResult:
    return ExperimentResult(
        name=name,
        accuracy=0.9,
        precision=0.8,
        recall=0.7,
        f1=f1,
        roc_auc=0.95,
        threshold=0.4,
        dataset="clean",
        epochs=1,
        parameters=100,
        note="заметка",
    )
