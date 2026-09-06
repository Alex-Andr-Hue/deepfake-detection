"""Готовит датасет: раскладка по классам, удаление шума, балансировка.

Собирает две версии одного и того же разбиения — шумную и очищенную, — чтобы
вклад предобработки можно было измерить, а не постулировать.

    python scripts/prepare_data.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deepfake.config import (  # noqa: E402
    CLEAN_DATASET,
    CLEAN_TEST_IMAGES,
    NOISY_DATASET,
    TEST_IMAGES,
    TRAIN_IMAGES,
    TRAIN_LABELS,
)
from deepfake.data import (  # noqa: E402
    balance_by_augmentation,
    build_split_dataset,
    summarize_dataset,
)
from deepfake.denoise import clean_folder  # noqa: E402


def check_raw_data() -> None:
    missing = [p for p in (TRAIN_IMAGES, TRAIN_LABELS) if not p.exists()]
    if missing:
        listing = "\n".join(f"  - {p}" for p in missing)
        raise SystemExit(
            f"не найдены исходные данные:\n{listing}\n\n"
            "Положите датасет в data/raw/ или укажите путь через DEEPFAKE_DATA."
        )


def prepare(split: bool, denoise: bool, balance: bool) -> None:
    check_raw_data()

    if split:
        for target in (NOISY_DATASET, CLEAN_DATASET):
            print(f"[split] {target}")
            counts = build_split_dataset(target, TRAIN_IMAGES, TRAIN_LABELS)
            print(f"        {counts}")

    if denoise:
        print(f"[denoise] {CLEAN_DATASET}")
        for sub in ("train_dataset", "test_dataset"):
            for class_name in ("0", "1"):
                folder = CLEAN_DATASET / sub / class_name
                count = clean_folder(folder)
                print(f"          {sub}/{class_name}: {count}")

        if TEST_IMAGES.exists() and not CLEAN_TEST_IMAGES.exists():
            print(f"[denoise] {TEST_IMAGES} -> {CLEAN_TEST_IMAGES}")
            print(f"          {clean_folder(TEST_IMAGES, CLEAN_TEST_IMAGES)}")

    if balance:
        for target in (NOISY_DATASET, CLEAN_DATASET):
            print(f"[balance] {target}")
            print(f"          {balance_by_augmentation(target / 'train_dataset')}")

    for target in (NOISY_DATASET, CLEAN_DATASET):
        if target.exists():
            print(f"{target.name}: {summarize_dataset(target)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="все три шага подряд")
    parser.add_argument("--split", action="store_true", help="разложить по классам")
    parser.add_argument("--denoise", action="store_true", help="удалить шум в чистой версии")
    parser.add_argument("--balance", action="store_true", help="выровнять классы аугментациями")
    args = parser.parse_args()

    if not any((args.all, args.split, args.denoise, args.balance)):
        parser.error("укажите --all или хотя бы один из шагов")

    prepare(
        split=args.all or args.split,
        denoise=args.all or args.denoise,
        balance=args.all or args.balance,
    )


if __name__ == "__main__":
    main()
