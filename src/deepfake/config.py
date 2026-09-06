"""Пути, гиперпараметры и фиксация случайности.

Все пути выводятся из корня репозитория, поэтому проект работает из любой
рабочей директории. Расположение данных переопределяется переменной окружения
``DEEPFAKE_DATA`` — удобно, когда датасет лежит на другом диске.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------- пути

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("DEEPFAKE_DATA", PROJECT_ROOT / "data"))

RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"

TRAIN_IMAGES = RAW_DIR / "train_images"
TEST_IMAGES = RAW_DIR / "test_images"
TRAIN_LABELS = RAW_DIR / "train_solution.csv"

#: исходные изображения, шум не тронут
NOISY_DATASET = PROCESSED_DIR / "noisy"
#: те же изображения после удаления шума
CLEAN_DATASET = PROCESSED_DIR / "clean"
#: соревновательный test_images после удаления шума
CLEAN_TEST_IMAGES = PROCESSED_DIR / "clean_test_images"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

COMPARISON_CSV = RESULTS_DIR / "model_comparison.csv"

# ---------------------------------------------------------------- константы

SEED = 42

#: размер hold-out выборки, стратифицированно по классам
HOLDOUT_FAKE = 1700
HOLDOUT_REAL = 8300

#: нормировка, посчитанная по обучающей выборке этого датасета
DATASET_MEAN = (0.5194, 0.4280, 0.3847)
DATASET_STD = (0.2861, 0.2640, 0.2636)

#: нормировка ImageNet — нужна моделям с предобученными весами
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class TrainingConfig:
    """Бюджет обучения. Одинаковый для всех моделей — иначе сравнение нечестное."""

    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_workers: int = 8
    image_size: int = 256

    def scaled(self, **overrides) -> TrainingConfig:
        """Копия конфига с изменёнными полями (для тестов и быстрых прогонов)."""
        return TrainingConfig(**{**self.__dict__, **overrides})


DEFAULT_TRAINING = TrainingConfig()


# ---------------------------------------------------------------- случайность


def set_seed(seed: int = SEED) -> None:
    """Фиксирует все источники случайности.

    Вызывается перед каждым экспериментом, чтобы разница между моделями
    объяснялась моделями, а не разной инициализацией и разным порядком батчей.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_generator(seed: int = SEED) -> torch.Generator:
    """Отдельный генератор для ``random_split``.

    Благодаря ему val и test состоят из одних и тех же изображений во всех
    экспериментах, и метрики разных моделей сравнимы между собой.
    """
    return torch.Generator().manual_seed(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def use_amp(device: torch.device | None = None) -> bool:
    """Mixed precision имеет смысл только на CUDA."""
    device = device or get_device()
    return device.type == "cuda"
