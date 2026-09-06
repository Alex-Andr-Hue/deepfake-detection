"""Цикл обучения и инференс.

Отбор чекпоинта идёт по F1, оптимизированному по порогу (``f1_over_thresholds``),
и в конце лучшие веса загружаются обратно в модель — после ``train_model``
в памяти гарантированно лежит именно тот чекпоинт, который записан на диск.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import DEFAULT_TRAINING, TrainingConfig, get_device, use_amp
from .metrics import classification_metrics, f1_over_thresholds


@dataclass
class EpochResult:
    loss: float
    labels: np.ndarray
    probs: np.ndarray

    @property
    def accuracy(self) -> float:
        return float(((self.probs >= 0.5).astype(int) == self.labels).mean())


@dataclass
class TrainingHistory:
    """История обучения — всё, что нужно для графиков и отчёта."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_f1: list[float] = field(default_factory=list)
    val_f1: list[float] = field(default_factory=list)
    val_tuned_f1: list[float] = field(default_factory=list)
    val_roc_auc: list[float] = field(default_factory=list)
    learning_rate: list[float] = field(default_factory=list)

    best_f1: float = 0.0
    best_epoch: int = 0
    best_threshold: float = 0.5

    def __len__(self) -> int:
        return len(self.train_loss)


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_function: nn.Module,
    optimizer: optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    device: torch.device | None = None,
    progress: bool = True,
) -> EpochResult:
    """Одна эпоха. Собирает вероятности класса 1, а не только жёсткие метки.

    ``optimizer=None`` переводит модель в режим оценки и отключает градиенты.
    """
    device = device or get_device()
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    all_probs, all_labels = [], []

    with torch.set_grad_enabled(is_train):
        for inputs, targets in tqdm(dataloader, leave=False, disable=not progress):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp(device)):
                logits = model(inputs)
                loss = loss_function(logits, targets)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item()
            all_probs.append(torch.softmax(logits.detach().float(), dim=1)[:, 1].cpu())
            all_labels.append(targets.cpu())

    return EpochResult(
        loss=total_loss / max(len(dataloader), 1),
        labels=torch.cat(all_labels).numpy(),
        probs=torch.cat(all_probs).numpy(),
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    checkpoint_path: Path,
    config: TrainingConfig = DEFAULT_TRAINING,
    loss_function: nn.Module | None = None,
    device: torch.device | None = None,
    progress: bool = True,
    verbose: bool = True,
) -> TrainingHistory:
    """Обучает модель, сохраняя лучший по val чекпоинт.

    Args:
        checkpoint_path: куда писать лучшие веса.
        config: бюджет обучения, одинаковый для всех моделей.

    Returns:
        История обучения; в ``best_threshold`` лежит порог, подобранный по val
        на лучшей эпохе — его и надо использовать при оценке на test.
    """
    device = device or get_device()
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    model.to(device)
    loss_function = loss_function or nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp(device))

    history = TrainingHistory()
    best_f1 = -1.0

    for epoch in range(config.epochs):
        history.learning_rate.append(optimizer.param_groups[0]["lr"])

        train_result = run_epoch(
            model, train_loader, loss_function, optimizer, scaler, device, progress
        )
        val_result = run_epoch(
            model, val_loader, loss_function, device=device, progress=progress
        )
        scheduler.step()

        train_metrics = classification_metrics(train_result.labels, train_result.probs)
        val_metrics = classification_metrics(val_result.labels, val_result.probs)
        tuned_f1, tuned_threshold = f1_over_thresholds(val_result.labels, val_result.probs)

        history.train_loss.append(train_result.loss)
        history.val_loss.append(val_result.loss)
        history.train_f1.append(train_metrics["f1"])
        history.val_f1.append(val_metrics["f1"])
        history.val_tuned_f1.append(tuned_f1)
        history.val_roc_auc.append(val_metrics["roc_auc"])

        if verbose:
            print(
                f"epoch {epoch + 1}/{config.epochs} | "
                f"train loss {train_result.loss:.4f} | "
                f"val loss {val_result.loss:.4f} "
                f"f1 {val_metrics['f1']:.4f} tuned {tuned_f1:.4f} "
                f"auc {val_metrics['roc_auc']:.4f}"
            )

        if tuned_f1 > best_f1:
            best_f1 = tuned_f1
            history.best_f1 = tuned_f1
            history.best_epoch = epoch + 1
            history.best_threshold = tuned_threshold
            torch.save(model.state_dict(), checkpoint_path)

    # возвращаем в модель лучшие веса, а не веса последней эпохи
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

    if verbose:
        print(
            f"лучший чекпоинт: эпоха {history.best_epoch}, "
            f"val F1 {history.best_f1:.4f} при пороге {history.best_threshold:.3f} "
            f"-> {checkpoint_path}"
        )

    return history


@torch.no_grad()
def predict_proba(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device | None = None,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает ``(истинные метки, вероятность класса 1)``.

    Работает при любом ``batch_size``.
    """
    device = device or get_device()
    model.to(device).eval()

    probs, targets = [], []
    for inputs, batch_targets in tqdm(dataloader, leave=False, disable=not progress):
        inputs = inputs.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp(device)):
            logits = model(inputs)
        probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu())
        targets.append(batch_targets)

    return torch.cat(targets).numpy(), torch.cat(probs).numpy()


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    threshold: float = 0.5,
    device: torch.device | None = None,
    progress: bool = True,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Метрики на выборке при заданном пороге.

    Returns:
        ``(метрики, метки, вероятности)``.
    """
    labels, probs = predict_proba(model, dataloader, device, progress)
    return classification_metrics(labels, probs, threshold), labels, probs
