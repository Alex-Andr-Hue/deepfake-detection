"""Оркестрация одного эксперимента.

Единая точка входа гарантирует, что все модели проходят через один и тот же
протокол: фиксированный seed, свежие веса, один и тот же split, одинаковый
бюджет обучения, порог и чекпоинт по val, метрики по test.
"""

from __future__ import annotations

from pathlib import Path

import torch.nn as nn

from .config import (
    CHECKPOINTS_DIR,
    DEFAULT_TRAINING,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SEED,
    TrainingConfig,
    set_seed,
)
from .data import make_loaders, make_transforms
from .metrics import REGISTRY, ExperimentRegistry, ExperimentResult
from .models import PRETRAINED_MODELS, build_model, count_parameters
from .training import TrainingHistory, evaluate, train_model


def slugify(text: str) -> str:
    """Имя файла чекпоинта из человекочитаемого названия эксперимента."""
    allowed = []
    for char in text.lower():
        allowed.append(char if char.isalnum() else "_")
    return "_".join(filter(None, "".join(allowed).split("_")))


def transforms_for(model_name: str, image_size: int):
    """Предобученным моделям нужна нормировка ImageNet, остальным — своя."""
    if model_name in PRETRAINED_MODELS:
        return make_transforms(IMAGENET_MEAN, IMAGENET_STD, image_size)
    return make_transforms(image_size=image_size)


def run_experiment(
    model_name: str,
    dataset_root: Path,
    label: str | None = None,
    note: str = "",
    config: TrainingConfig = DEFAULT_TRAINING,
    registry: ExperimentRegistry = REGISTRY,
    checkpoints_dir: Path = CHECKPOINTS_DIR,
    tune_threshold: bool = True,
    seed: int = SEED,
    model: nn.Module | None = None,
    progress: bool = True,
    verbose: bool = True,
) -> tuple[nn.Module, TrainingHistory, ExperimentResult]:
    """Прогоняет один эксперимент от начала до конца.

    Args:
        model_name: ключ из ``MODEL_FACTORIES``.
        dataset_root: папка с ``train_dataset`` и ``test_dataset``.
        label: имя строки в итоговой таблице; по умолчанию — ``model_name``.
        model: готовая модель вместо создания новой; нужно только для
            многостадийных сценариев вроде «сначала шум, потом чистые».

    Returns:
        ``(модель, история, результат)``. В модели лежат лучшие по val веса.
    """
    label = label or model_name
    if verbose:
        print(f"===== {label} =====")

    set_seed(seed)
    if model is None:
        model = build_model(model_name, seed)

    train_transform, eval_transform = transforms_for(model_name, config.image_size)
    train_loader, val_loader, test_loader = make_loaders(
        dataset_root,
        config,
        train_transform=train_transform,
        eval_transform=eval_transform,
        seed=seed,
    )

    history = train_model(
        model,
        train_loader,
        val_loader,
        checkpoint_path=Path(checkpoints_dir) / f"{slugify(label)}.pth",
        config=config,
        progress=progress,
        verbose=verbose,
    )

    # порог уже подобран по val для лучшей эпохи — незачем гонять валидацию ещё раз
    threshold = history.best_threshold if tune_threshold else 0.5
    metrics, _, _ = evaluate(model, test_loader, threshold, progress=progress)

    result = ExperimentResult.from_metrics(
        label,
        metrics,
        dataset=Path(dataset_root).name,
        epochs=config.epochs,
        parameters=count_parameters(model),
        note=note,
    )
    registry.add(result)

    if verbose:
        print(
            f"test: f1 {metrics['f1']:.4f} | roc_auc {metrics['roc_auc']:.4f} | "
            f"accuracy {metrics['accuracy']:.4f} | порог {threshold:.3f}"
        )

    return model, history, result
