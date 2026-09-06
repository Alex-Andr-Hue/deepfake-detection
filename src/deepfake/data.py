"""Подготовка датасета, балансировка классов и загрузчики.

Ключевое свойство: hold-out отбирается детерминированно, а деление hold-out на
val и test использует генератор с фиксированным seed. Поэтому все модели во всех
экспериментах видят одни и те же val и test, и их метрики можно ставить в одну
таблицу.
"""

from __future__ import annotations

import csv
import random
import shutil
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .config import (
    DATASET_MEAN,
    DATASET_STD,
    HOLDOUT_FAKE,
    HOLDOUT_REAL,
    SEED,
    TrainingConfig,
    split_generator,
)

CLASS_REAL = "0"
CLASS_FAKE = "1"
CLASS_NAMES = ("real", "fake")

#: префиксы, которыми помечаются сгенерированные аугментации
AUGMENTATION_PREFIXES = ("flip_", "affine_", "light_", "all_")


# ------------------------------------------------------------------ раскладка


def build_split_dataset(
    target_dir: Path,
    images_dir: Path,
    labels_csv: Path,
    holdout_fake: int = HOLDOUT_FAKE,
    holdout_real: int = HOLDOUT_REAL,
    seed: int = SEED,
) -> dict[str, int]:
    """Раскладывает изображения по папкам ``train_dataset`` / ``test_dataset``.

    Hold-out отбирается стратифицированно и детерминированно: ``random.Random``
    с фиксированным seed по отсортированному списку id. Благодаря этому шумная
    и очищенная версии датасета содержат ровно одни и те же изображения в train
    и в hold-out, и разница в метриках объясняется только предобработкой.

    Функция не хранит состояния между вызовами, поэтому повторные вызовы для
    разных ``target_dir`` полностью независимы.

    Args:
        target_dir: куда складывать результат.
        images_dir: папка с исходными изображениями ``<id>.jpg``.
        labels_csv: csv без заголовка, колонки ``id,label``.

    Returns:
        Словарь с числом изображений в каждой части.
    """
    target_dir = Path(target_dir)
    if target_dir.exists():
        return summarize_dataset(target_dir)

    by_class: dict[str, list[str]] = {CLASS_REAL: [], CLASS_FAKE: []}
    with open(labels_csv) as handle:
        for image_id, is_fake in csv.reader(handle):
            by_class[CLASS_FAKE if int(is_fake) else CLASS_REAL].append(image_id)

    rng = random.Random(seed)
    holdout: set[str] = set()
    for class_name, need in ((CLASS_REAL, holdout_real), (CLASS_FAKE, holdout_fake)):
        ids = sorted(by_class[class_name], key=int)
        holdout.update(rng.sample(ids, min(need, len(ids))))

    for split in ("train_dataset", "test_dataset"):
        for class_name in (CLASS_REAL, CLASS_FAKE):
            (target_dir / split / class_name).mkdir(parents=True, exist_ok=True)

    for class_name, ids in by_class.items():
        for image_id in ids:
            split = "test_dataset" if image_id in holdout else "train_dataset"
            shutil.copy(
                Path(images_dir) / f"{image_id}.jpg",
                target_dir / split / class_name,
            )

    return summarize_dataset(target_dir)


def summarize_dataset(root: Path) -> dict[str, int]:
    """Считает изображения в каждой части датасета."""
    root = Path(root)
    counts = {}
    for split in ("train_dataset", "test_dataset"):
        for class_name, human in ((CLASS_REAL, "real"), (CLASS_FAKE, "fake")):
            folder = root / split / class_name
            counts[f"{split.split('_')[0]}_{human}"] = (
                sum(1 for _ in folder.iterdir()) if folder.exists() else 0
            )
    return counts


# ------------------------------------------------------------- балансировка


def _augmentations() -> dict[str, transforms.Compose]:
    flip = transforms.RandomHorizontalFlip(p=1.0)
    affine = transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1))
    light = transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25)
    return {
        "flip_": flip,
        "affine_": affine,
        "light_": light,
        "all_": transforms.Compose([flip, affine, light]),
    }


def balance_by_augmentation(
    train_dir: Path,
    seed: int = SEED,
) -> dict[str, int]:
    """Доводит меньший класс до размера большего аугментациями.

    Из трёх способов бороться с дисбалансом — физическая аугментация,
    ``WeightedRandomSampler`` и веса классов в лоссе — используется только этот.
    Они взаимозаменяемы, и накладывать их друг на друга вредно: после
    выравнивания классов сэмплер становится пустой операцией, а веса,
    посчитанные по исходному распределению, перекашивают обучение в обратную
    сторону.

    Аугментации пишутся только в train — hold-out не трогается, утечки нет.

    Returns:
        Итоговое число изображений в каждом классе.
    """
    train_dir = Path(train_dir)
    real_dir = train_dir / CLASS_REAL
    fake_dir = train_dir / CLASS_FAKE

    real_count = sum(1 for _ in real_dir.iterdir())
    existing = {p.name for p in fake_dir.iterdir()}
    originals = [
        name for name in existing if not name.startswith(AUGMENTATION_PREFIXES)
    ]

    augmentations = _augmentations()
    rng = random.Random(seed)

    # по одной копии каждого вида аугментации на каждый оригинал
    for name in originals:
        image = Image.open(fake_dir / name).convert("RGB")
        for prefix, transform in augmentations.items():
            new_name = prefix + name
            if new_name not in existing:
                transform(image).save(fake_dir / new_name)
                existing.add(new_name)

    # подгоняем ровно под размер большего класса
    surplus = len(existing) - real_count
    if surplus > 0:
        removable = [n for n in existing if n.startswith(AUGMENTATION_PREFIXES)]
        for name in rng.sample(removable, min(surplus, len(removable))):
            (fake_dir / name).unlink()
            existing.discard(name)
    elif surplus < 0:
        pool = sorted(existing)
        for index in range(-surplus):
            source = rng.choice(pool)
            prefix, transform = rng.choice(list(augmentations.items()))
            image = Image.open(fake_dir / source).convert("RGB")
            transform(image).save(fake_dir / f"{prefix}{index}_{source}")

    return {
        "real": real_count,
        "fake": sum(1 for _ in fake_dir.iterdir()),
    }


# -------------------------------------------------------------- трансформации


def make_transforms(
    mean: tuple[float, ...] = DATASET_MEAN,
    std: tuple[float, ...] = DATASET_STD,
    image_size: int = 256,
) -> tuple[transforms.Compose, transforms.Compose]:
    """Возвращает пару (train, eval).

    Аугментации применяются только к train. Для val и test — исключительно
    детерминированная нормировка, иначе метрики шумят от запуска к запуску.
    """
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transform, eval_transform


# ------------------------------------------------------------------ загрузчики


class ImageFolderFlat(Dataset):
    """Плоская папка изображений, имя файла — это id. Для инференса."""

    def __init__(self, folder: Path, transform: transforms.Compose):
        self.folder = Path(folder)
        self.transform = transform
        self.files = sorted(
            (p for p in self.folder.iterdir() if p.is_file()),
            key=lambda p: int(p.stem),
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path = self.files[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), int(path.stem)


def make_loaders(
    root: Path,
    config: TrainingConfig,
    train_transform: transforms.Compose | None = None,
    eval_transform: transforms.Compose | None = None,
    use_sampler: bool = False,
    seed: int = SEED,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Собирает train/val/test загрузчики для одного датасета.

    Hold-out делится пополам на val и test генератором с фиксированным seed,
    поэтому состав выборок одинаков во всех экспериментах.

    Args:
        use_sampler: включать ``WeightedRandomSampler``. Нужен, только если
            train-папка не выровнена физически: после ``balance_by_augmentation``
            это пустая операция.
    """
    root = Path(root)
    if train_transform is None or eval_transform is None:
        default_train, default_eval = make_transforms(image_size=config.image_size)
        train_transform = train_transform or default_train
        eval_transform = eval_transform or default_eval

    train_dataset = ImageFolder(str(root / "train_dataset"), transform=train_transform)
    holdout = ImageFolder(str(root / "test_dataset"), transform=eval_transform)
    val_dataset, test_dataset = random_split(
        holdout, [0.5, 0.5], generator=split_generator(seed)
    )

    if use_sampler:
        counts = Counter(train_dataset.targets)
        weights = [1.0 / counts[target] for target in train_dataset.targets]
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
            generator=split_generator(seed),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=config.num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
