"""Метрики классификации, подбор порога и реестр результатов.

Два принципиальных момента:

* **ROC-AUC считается по вероятностям.** AUC от жёстких меток — это не ROC-AUC,
  а балансированная точность; на перекошенных классах разница огромна.
* **Порог подбирается, а не берётся равным 0.5.** ``argmax`` эквивалентен порогу
  0.5, но при дисбалансе 5:1 оптимум по F1 обычно лежит заметно ниже.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_COLUMNS = ("accuracy", "precision", "recall", "f1", "roc_auc", "threshold")


def f1_over_thresholds(labels: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    """Лучший достижимый F1 и порог, на котором он достигается.

    Метрика, не зависящая от произвольного порога 0.5. По ней честно отбирать
    чекпоинт: иначе модель с хорошим ранжированием, но смещённой калибровкой,
    получает F1 = 0 и выглядит бесполезной, а если F1 нулевой на всех эпохах —
    чекпоинт выбирается фактически случайно.

    Returns:
        ``(лучший F1, порог)``.
    """
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    denominator = precision + recall
    f1_grid = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(precision),
        where=denominator > 0,
    )
    # последний элемент precision_recall_curve не соответствует порогу
    best = int(np.argmax(f1_grid[:-1]))
    return float(f1_grid[best]), float(thresholds[best])


def classification_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Полный набор метрик при заданном пороге."""
    predicts = (probs >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, predicts)),
        "precision": float(precision_score(labels, predicts, zero_division=0)),
        "recall": float(recall_score(labels, predicts, zero_division=0)),
        "f1": float(f1_score(labels, predicts, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probs)),
        "threshold": float(threshold),
    }


def confusion(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return confusion_matrix(labels, (probs >= threshold).astype(int))


@dataclass
class ExperimentResult:
    """Результат одного эксперимента."""

    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    threshold: float
    dataset: str = ""
    epochs: int = 0
    parameters: int = 0
    note: str = ""

    @classmethod
    def from_metrics(cls, name: str, metrics: dict[str, float], **extra) -> ExperimentResult:
        return cls(name=name, **{k: metrics[k] for k in METRIC_COLUMNS}, **extra)


@dataclass
class ExperimentRegistry:
    """Накопитель результатов, из которого собирается итоговая таблица."""

    results: dict[str, ExperimentResult] = field(default_factory=dict)

    def add(self, result: ExperimentResult) -> None:
        self.results[result.name] = result

    def to_frame(self) -> pd.DataFrame:
        """Сводная таблица, отсортированная по F1."""
        if not self.results:
            return pd.DataFrame(columns=["name", *METRIC_COLUMNS])

        frame = pd.DataFrame([asdict(r) for r in self.results.values()])
        frame[list(METRIC_COLUMNS)] = frame[list(METRIC_COLUMNS)].round(4)
        return frame.sort_values("f1", ascending=False).set_index("name")

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(path)

    @classmethod
    def load(cls, path: Path) -> ExperimentRegistry:
        registry = cls()
        path = Path(path)
        if not path.exists():
            return registry

        frame = pd.read_csv(path)
        for row in frame.to_dict(orient="records"):
            registry.add(ExperimentResult(**row))
        return registry

    def save_json(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {name: asdict(r) for name, r in self.results.items()}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


#: общий реестр, в который пишут скрипты экспериментов
REGISTRY = ExperimentRegistry()
