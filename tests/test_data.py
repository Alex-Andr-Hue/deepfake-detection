"""Данные: детерминированность разбиения, балансировка, загрузчики."""

from __future__ import annotations

import csv

import numpy as np
import pytest
from PIL import Image

from deepfake.config import DEFAULT_TRAINING
from deepfake.data import (
    ImageFolderFlat,
    balance_by_augmentation,
    build_split_dataset,
    make_loaders,
    make_transforms,
    summarize_dataset,
)


@pytest.fixture
def raw_data(tmp_path):
    """Сырые изображения и csv с метками, как в исходном датасете."""
    rng = np.random.default_rng(11)
    images = tmp_path / "train_images"
    images.mkdir()

    labels = []
    for image_id in range(60):
        array = rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)
        Image.fromarray(array).save(images / f"{image_id}.jpg")
        labels.append((image_id, 1 if image_id % 5 == 0 else 0))

    csv_path = tmp_path / "labels.csv"
    with open(csv_path, "w", newline="") as handle:
        csv.writer(handle).writerows(labels)

    return images, csv_path


def test_split_is_deterministic(raw_data, tmp_path):
    """Два независимых вызова дают одинаковый hold-out.

    Это то, что делает шумную и очищенную версии датасета сравнимыми: они
    содержат одни и те же изображения в train и в hold-out.
    """
    images, labels = raw_data

    first = tmp_path / "first"
    second = tmp_path / "second"
    build_split_dataset(first, images, labels, holdout_fake=4, holdout_real=8)
    build_split_dataset(second, images, labels, holdout_fake=4, holdout_real=8)

    for split in ("train_dataset", "test_dataset"):
        for class_name in ("0", "1"):
            names_first = sorted(p.name for p in (first / split / class_name).iterdir())
            names_second = sorted(p.name for p in (second / split / class_name).iterdir())
            assert names_first == names_second


def test_split_has_no_overlap(raw_data, tmp_path):
    """Ни одно изображение не попадает и в train, и в hold-out."""
    images, labels = raw_data
    root = tmp_path / "dataset"
    build_split_dataset(root, images, labels, holdout_fake=4, holdout_real=8)

    train_names, holdout_names = set(), set()
    for class_name in ("0", "1"):
        train_names |= {p.name for p in (root / "train_dataset" / class_name).iterdir()}
        holdout_names |= {p.name for p in (root / "test_dataset" / class_name).iterdir()}

    assert not (train_names & holdout_names)


def test_split_respects_holdout_sizes(raw_data, tmp_path):
    images, labels = raw_data
    root = tmp_path / "dataset"
    counts = build_split_dataset(root, images, labels, holdout_fake=4, holdout_real=8)

    assert counts["test_fake"] == 4
    assert counts["test_real"] == 8
    assert counts["train_fake"] + counts["test_fake"] == 12
    assert counts["train_real"] + counts["test_real"] == 48


def test_split_is_idempotent(raw_data, tmp_path):
    """Повторный вызов на существующей папке ничего не ломает и не дублирует."""
    images, labels = raw_data
    root = tmp_path / "dataset"

    first = build_split_dataset(root, images, labels, holdout_fake=4, holdout_real=8)
    second = build_split_dataset(root, images, labels, holdout_fake=4, holdout_real=8)

    assert first == second


def test_balance_equalises_classes(raw_data, tmp_path):
    images, labels = raw_data
    root = tmp_path / "dataset"
    build_split_dataset(root, images, labels, holdout_fake=4, holdout_real=8)

    counts = balance_by_augmentation(root / "train_dataset")

    assert counts["fake"] == counts["real"]


def test_balance_does_not_touch_holdout(raw_data, tmp_path):
    """Аугментации пишутся только в train — иначе была бы утечка в оценку."""
    images, labels = raw_data
    root = tmp_path / "dataset"
    build_split_dataset(root, images, labels, holdout_fake=4, holdout_real=8)

    before = summarize_dataset(root)
    balance_by_augmentation(root / "train_dataset")
    after = summarize_dataset(root)

    assert before["test_fake"] == after["test_fake"]
    assert before["test_real"] == after["test_real"]


def test_loaders_split_holdout_identically(tiny_dataset):
    """val и test одинаковы между вызовами — иначе модели несравнимы."""
    config = DEFAULT_TRAINING.scaled(batch_size=4, num_workers=0, image_size=64)

    def holdout_targets():
        _, val_loader, test_loader = make_loaders(tiny_dataset, config)
        return (
            [int(t) for _, targets in val_loader for t in targets],
            [int(t) for _, targets in test_loader for t in targets],
        )

    assert holdout_targets() == holdout_targets()


def test_loaders_produce_expected_shapes(tiny_dataset):
    config = DEFAULT_TRAINING.scaled(batch_size=4, num_workers=0, image_size=64)
    train_loader, val_loader, _ = make_loaders(tiny_dataset, config)

    inputs, targets = next(iter(train_loader))
    assert inputs.shape[1:] == (3, 64, 64)
    assert targets.ndim == 1

    assert len(val_loader.dataset) + len(_.dataset) == 16


def test_eval_transform_is_deterministic():
    """На val и test аугментаций быть не должно — иначе метрики шумят."""
    _, eval_transform = make_transforms(image_size=32)
    image = Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8))

    assert eval_transform(image).allclose(eval_transform(image))


def test_flat_dataset_sorts_numerically(flat_images):
    """Имена файлов — числа: сортировка должна быть числовой, а не строковой."""
    _, eval_transform = make_transforms(image_size=32)
    dataset = ImageFolderFlat(flat_images, eval_transform)

    ids = [dataset[i][1] for i in range(len(dataset))]
    assert ids == sorted(ids)
