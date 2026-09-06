"""Общие фикстуры: синтетические изображения и крошечный датасет на диске."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def smooth_image() -> np.ndarray:
    """Гладкий градиент без шума — фильтр не должен на нём срабатывать."""
    rows, cols = np.mgrid[0:64, 0:64]
    plane = (rows * 0.8 + cols * 0.5 + 40).astype(np.uint8)
    return np.stack([plane] * 3, axis=-1)


@pytest.fixture
def salt_pepper_coordinates() -> list[tuple[int, int]]:
    rng = np.random.default_rng(0)
    return sorted({(int(rng.integers(2, 62)), int(rng.integers(2, 62))) for _ in range(60)})


@pytest.fixture
def noisy_image(smooth_image, salt_pepper_coordinates) -> np.ndarray:
    """Тот же градиент с выбитыми пикселями."""
    image = smooth_image.copy()
    for index, (row, col) in enumerate(salt_pepper_coordinates):
        image[row, col] = 255 if index % 2 else 0
    return image


@pytest.fixture
def tiny_dataset(tmp_path) -> Path:
    """Датасет из 40 картинок в структуре ImageFolder.

    Класс 1 отличается от класса 0 уровнем яркости, так что задача решаема и
    метрики осмысленны, но обучение занимает секунды.
    """
    rng = np.random.default_rng(7)
    root = tmp_path / "dataset"

    for split, per_class in (("train_dataset", 12), ("test_dataset", 8)):
        for class_name in ("0", "1"):
            folder = root / split / class_name
            folder.mkdir(parents=True)
            for index in range(per_class):
                base = rng.integers(90, 130, (32, 32, 3)).astype(np.float32)
                if class_name == "1":
                    base += 40
                array = np.clip(base, 0, 255).astype(np.uint8)
                Image.fromarray(array).resize((64, 64)).save(folder / f"{index}.jpg")

    return root


@pytest.fixture
def flat_images(tmp_path) -> Path:
    """Плоская папка `<id>.jpg` для инференса."""
    rng = np.random.default_rng(3)
    folder = tmp_path / "flat"
    folder.mkdir()
    for index in range(6):
        array = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(array).save(folder / f"{index}.jpg")
    return folder


@pytest.fixture
def labelled_probabilities() -> tuple[np.ndarray, np.ndarray]:
    """Перекошенные классы и вероятности, смещённые вниз.

    Ровно тот случай, ради которого подбирается порог: ранжирование хорошее,
    но при пороге 0.5 почти ничего не попадает в положительный класс.
    """
    rng = np.random.default_rng(1)
    labels = np.concatenate([np.zeros(900, int), np.ones(100, int)])
    probs = np.concatenate([rng.beta(2, 12, 900), rng.beta(5, 8, 100)])
    return labels, probs
