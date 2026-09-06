"""Простая свёрточная сеть — точка отсчёта для остальных архитектур."""

from __future__ import annotations

import torch.nn as nn


def make_cnn(num_classes: int = 2, dropout: float = 0.3) -> nn.Module:
    """Пять свёрточных блоков и классификатор.

    Между двумя ``Linear`` стоит ``ReLU``: без нелинейности пара
    ``Linear(4096, 512) -> Linear(512, 2)`` математически эквивалентна одному
    ``Linear(4096, 2)``, то есть скрытый слой не добавляет выразительности.

    Рассчитана на вход 256x256: пять ``MaxPool2d(2)`` дают карту 8x8.
    """
    return nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),   # 128 x 128

        nn.Conv2d(8, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),   # 64 x 64

        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),   # 32 x 32

        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),   # 16 x 16

        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),   # 8 x 8

        nn.Flatten(),
        nn.Linear(8 * 8 * 64, 512),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(512, num_classes),
    )
