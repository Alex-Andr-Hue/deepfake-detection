"""Фильтр шума: точечность замены и корректность среднего по соседям."""

from __future__ import annotations

import numpy as np
import pytest

from deepfake.denoise import clean_folder, neighbour_mean, noise_ratio, remove_noise


def test_neighbour_mean_excludes_centre():
    """Центральный пиксель не входит в собственное среднее.

    Если бы входил, выброс 250 на фоне 10 сдвинул бы среднее до 36.7 и
    замаскировал сам себя.
    """
    image = np.full((5, 5, 3), 10, dtype=np.uint8)
    image[2, 2] = 250

    mean = neighbour_mean(image)

    assert mean[2, 2, 0] == pytest.approx(10.0, abs=1e-4)


def test_neighbour_mean_on_constant_image():
    image = np.full((8, 8, 3), 77, dtype=np.uint8)
    assert neighbour_mean(image) == pytest.approx(77.0, abs=1e-4)


def test_rejects_even_kernel():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="нечётным"):
        neighbour_mean(image, kernel=4)


def test_removes_salt_and_pepper(noisy_image, salt_pepper_coordinates):
    cleaned = remove_noise(noisy_image)

    repaired = sum(
        1
        for row, col in salt_pepper_coordinates
        if not np.array_equal(cleaned[row, col], noisy_image[row, col])
    )
    assert repaired >= 0.9 * len(salt_pepper_coordinates)


def test_leaves_clean_pixels_alone(noisy_image, salt_pepper_coordinates):
    """Замена точечная: пиксели вне шума не трогаются."""
    cleaned = remove_noise(noisy_image)

    untouched = np.ones(noisy_image.shape[:2], dtype=bool)
    for row, col in salt_pepper_coordinates:
        untouched[row, col] = False

    collateral = (cleaned != noisy_image).any(axis=2) & untouched
    assert collateral.sum() <= 0.01 * untouched.size


def test_smooth_image_survives_untouched(smooth_image):
    """На изображении без шума фильтр не должен менять ничего."""
    assert np.array_equal(remove_noise(smooth_image), smooth_image)


def test_preserves_dtype_and_shape(noisy_image):
    cleaned = remove_noise(noisy_image)
    assert cleaned.dtype == np.uint8
    assert cleaned.shape == noisy_image.shape


def test_noise_ratio_reflects_damage(smooth_image, noisy_image):
    assert noise_ratio(smooth_image) == 0.0
    assert noise_ratio(noisy_image) > 0.0


def test_higher_threshold_removes_less(noisy_image):
    assert noise_ratio(noisy_image, threshold=20) >= noise_ratio(noisy_image, threshold=80)


def test_clean_folder_writes_to_destination(flat_images, tmp_path):
    destination = tmp_path / "cleaned"
    count = clean_folder(flat_images, destination)

    assert count == 6
    assert sorted(p.name for p in destination.iterdir()) == sorted(
        p.name for p in flat_images.iterdir()
    )
