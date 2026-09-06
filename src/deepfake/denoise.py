"""Удаление импульсного шума заменой выбитых пикселей средним по соседям.

Денойзинг здесь — вспомогательный шаг предобработки, а не предмет исследования.

Замена принципиально **точечная**: трогаются только пиксели, которые реально
выбиваются из своей окрестности. Сглаживать изображение целиком нельзя —
артефакты генерации, по которым и отличается дипфейк, лежат в том же
высокочастотном диапазоне, что и шум, и сплошной фильтр стирает полезный
признак вместе с помехой.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

#: порог: на сколько уровней пиксель должен отличаться от соседей, чтобы
#: считаться шумом. 40 подобрано визуально по обучающей выборке.
DEFAULT_THRESHOLD = 40
DEFAULT_KERNEL = 3


def neighbour_mean(image: np.ndarray, kernel: int = DEFAULT_KERNEL) -> np.ndarray:
    """Среднее по соседним пикселям, без учёта самого пикселя.

    ``boxFilter`` с ``normalize=False`` даёт сумму по окну ``kernel x kernel``
    вместе с центром; вычитаем центр и делим на число соседей.

    Исключать центр обязательно: иначе выбитый пиксель входит в собственное
    среднее, тянет его на себя и маскирует сам себя — при ядре 3x3 выброс
    сдвигает среднее на 1/9 своей амплитуды.

    Args:
        image: изображение HxWxC в uint8.
        kernel: размер окна, нечётный.

    Returns:
        Массив float32 той же формы со средним по соседям.
    """
    if kernel % 2 == 0:
        raise ValueError(f"размер окна должен быть нечётным, получено {kernel}")

    window_sum = cv2.boxFilter(
        image.astype(np.float32),
        ddepth=cv2.CV_32F,
        ksize=(kernel, kernel),
        normalize=False,
    )
    return (window_sum - image.astype(np.float32)) / (kernel * kernel - 1)


def remove_noise(
    image: np.ndarray,
    threshold: int = DEFAULT_THRESHOLD,
    kernel: int = DEFAULT_KERNEL,
) -> np.ndarray:
    """Заменяет шумные пиксели средним по их соседям.

    Пиксель считается шумом, если хотя бы в одном канале отличается от среднего
    по соседям больше чем на ``threshold``. Остальные пиксели не изменяются.

    Args:
        image: изображение HxWxC в uint8 (RGB или BGR — не важно).
        threshold: порог срабатывания.
        kernel: размер окрестности.

    Returns:
        Очищенное изображение uint8 той же формы.
    """
    mean = neighbour_mean(image, kernel)
    difference = np.abs(image.astype(np.float32) - mean)

    noisy_mask = difference.max(axis=2) > threshold

    output = image.copy()
    output[noisy_mask] = np.clip(mean, 0, 255).astype(np.uint8)[noisy_mask]
    return output


def noise_ratio(
    image: np.ndarray,
    threshold: int = DEFAULT_THRESHOLD,
    kernel: int = DEFAULT_KERNEL,
) -> float:
    """Доля пикселей, которые фильтр считает шумом. Полезно для подбора порога."""
    mean = neighbour_mean(image, kernel)
    difference = np.abs(image.astype(np.float32) - mean)
    return float((difference.max(axis=2) > threshold).mean())


def clean_image_file(path: Path, destination: Path | None = None, **kwargs) -> None:
    """Читает изображение, чистит и сохраняет (по умолчанию — на место)."""
    image = cv2.cvtColor(cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    Image.fromarray(remove_noise(image, **kwargs)).save(destination or path)


def clean_folder(source: Path, destination: Path | None = None, **kwargs) -> int:
    """Чистит все изображения в папке.

    Args:
        source: папка с изображениями.
        destination: куда складывать; ``None`` — перезаписать на месте.

    Returns:
        Число обработанных файлов.
    """
    source = Path(source)
    if destination is not None:
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in source.iterdir() if p.is_file())
    for path in files:
        target = None if destination is None else destination / path.name
        clean_image_file(path, target, **kwargs)

    return len(files)
