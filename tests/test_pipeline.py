"""Сквозной прогон: обучение, отбор чекпоинта, оценка, реестр."""

from __future__ import annotations

import pytest
import torch

from deepfake.config import DEFAULT_TRAINING
from deepfake.data import make_loaders
from deepfake.experiment import run_experiment, slugify
from deepfake.metrics import ExperimentRegistry
from deepfake.training import evaluate, predict_proba, train_model


@pytest.fixture
def config():
    return DEFAULT_TRAINING.scaled(epochs=2, batch_size=4, num_workers=0, image_size=64)


@pytest.fixture
def small_model():
    """Компактная сеть под вход 64x64 — CNN из реестра рассчитана на 256x256."""
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(8, 2),
    )


def test_train_model_records_history(tiny_dataset, config, small_model, tmp_path):
    train_loader, val_loader, _ = make_loaders(tiny_dataset, config)

    history = train_model(
        small_model, train_loader, val_loader,
        checkpoint_path=tmp_path / "model.pth",
        config=config, progress=False, verbose=False,
    )

    assert len(history) == 2
    assert len(history.val_tuned_f1) == 2
    assert 1 <= history.best_epoch <= 2
    assert 0.0 < history.best_threshold < 1.0


def test_checkpoint_is_written(tiny_dataset, config, small_model, tmp_path):
    train_loader, val_loader, _ = make_loaders(tiny_dataset, config)
    checkpoint = tmp_path / "nested" / "model.pth"

    train_model(
        small_model, train_loader, val_loader,
        checkpoint_path=checkpoint, config=config, progress=False, verbose=False,
    )

    assert checkpoint.exists()


def test_best_weights_are_restored(tiny_dataset, config, small_model, tmp_path):
    """После обучения в памяти лежит лучший чекпоинт, а не последняя эпоха."""
    train_loader, val_loader, _ = make_loaders(tiny_dataset, config)
    checkpoint = tmp_path / "model.pth"

    train_model(
        small_model, train_loader, val_loader,
        checkpoint_path=checkpoint, config=config, progress=False, verbose=False,
    )

    saved = torch.load(checkpoint, weights_only=True)
    for key, value in small_model.state_dict().items():
        assert torch.equal(value.cpu(), saved[key]), key


def test_predict_proba_covers_whole_dataset(tiny_dataset, config, small_model):
    _, _, test_loader = make_loaders(tiny_dataset, config)

    labels, probs = predict_proba(small_model, test_loader, progress=False)

    assert len(labels) == len(test_loader.dataset)
    assert labels.shape == probs.shape
    assert ((probs >= 0.0) & (probs <= 1.0)).all()


def test_predict_proba_is_batch_size_independent(tiny_dataset, small_model):
    """Метрики не должны зависеть от размера батча.

    Регрессионный тест: `argmax` без `dim=1` даёт правильный ответ только при
    batch_size = 1 и молча ломается на больших батчах.
    """
    small_model.eval()

    def probabilities(batch_size):
        config = DEFAULT_TRAINING.scaled(
            epochs=1, batch_size=batch_size, num_workers=0, image_size=64
        )
        _, _, test_loader = make_loaders(tiny_dataset, config)
        labels, probs = predict_proba(small_model, test_loader, progress=False)
        return labels, probs

    labels_one, probs_one = probabilities(1)
    labels_many, probs_many = probabilities(8)

    assert (labels_one == labels_many).all()
    assert probs_one == pytest.approx(probs_many, abs=1e-5)


def test_evaluate_returns_metrics(tiny_dataset, config, small_model):
    _, _, test_loader = make_loaders(tiny_dataset, config)

    metrics, labels, probs = evaluate(small_model, test_loader, threshold=0.5, progress=False)

    assert set(metrics) >= {"accuracy", "precision", "recall", "f1", "roc_auc", "threshold"}
    assert len(labels) == len(probs)


def test_run_experiment_populates_registry(tiny_dataset, config, tmp_path, monkeypatch):
    """Полный цикл через публичную точку входа."""
    registry = ExperimentRegistry()

    monkeypatch.setattr(
        "deepfake.experiment.build_model",
        lambda name, seed=0: torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(8, 2),
        ),
    )

    _, history, result = run_experiment(
        "CNN", tiny_dataset, label="Тестовая модель", note="прогон",
        config=config, registry=registry, checkpoints_dir=tmp_path,
        progress=False, verbose=False,
    )

    assert result.name == "Тестовая модель"
    assert result.epochs == 2
    assert result.parameters > 0
    assert result.threshold == pytest.approx(history.best_threshold)
    assert "Тестовая модель" in registry.results
    assert not registry.to_frame().empty


def test_run_experiment_reuses_supplied_model(tiny_dataset, config, tmp_path, small_model):
    """Многостадийные сценарии дообучают переданную модель, а не создают новую."""
    registry = ExperimentRegistry()

    returned, _, _ = run_experiment(
        "CNN", tiny_dataset, label="Стадия 2",
        config=config, registry=registry, checkpoints_dir=tmp_path,
        model=small_model, progress=False, verbose=False,
    )

    assert returned is small_model


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("InceptionV1", "inceptionv1"),
        ("ResNet18 (ImageNet)", "resnet18_imagenet"),
        ("InceptionV1+Freq", "inceptionv1_freq"),
    ],
)
def test_slugify(text, expected):
    assert slugify(text) == expected
