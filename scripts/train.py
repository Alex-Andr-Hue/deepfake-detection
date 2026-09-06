"""Запускает эксперименты по единому протоколу и пишет итоговую таблицу.

    python scripts/train.py --all                  # весь набор сравнений
    python scripts/train.py --models ResNet18      # только одна модель
    python scripts/train.py --all --epochs 5       # быстрый прогон
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deepfake.config import (  # noqa: E402
    CLEAN_DATASET,
    COMPARISON_CSV,
    DEFAULT_TRAINING,
    NOISY_DATASET,
    RESULTS_DIR,
)
from deepfake.experiment import run_experiment  # noqa: E402
from deepfake.metrics import REGISTRY  # noqa: E402
from deepfake.models import MODEL_FACTORIES  # noqa: E402

#: базовое сравнение архитектур: все на очищенных данных, всё остальное одинаково
ARCHITECTURE_SWEEP = list(MODEL_FACTORIES)


def run_noise_ablation(config, progress: bool) -> None:
    """Вклад очистки: та же модель на шумной версии того же разбиения."""
    run_experiment(
        "InceptionV1",
        NOISY_DATASET,
        label="InceptionV1 (шумные)",
        note="без удаления шума",
        config=config,
        progress=progress,
    )


def run_two_stage(config, progress: bool) -> None:
    """Шум как предобучение: сначала шумные данные, потом очищенные.

    Утечки нет: hold-out обеих версий состоит из одних и тех же id, поэтому
    первая стадия не видит тестовых изображений второй.
    """
    model, _, _ = run_experiment(
        "InceptionV1",
        NOISY_DATASET,
        label="InceptionV1 (стадия 1: шум)",
        note="промежуточная стадия",
        config=config,
        progress=progress,
    )
    REGISTRY.results.pop("InceptionV1 (стадия 1: шум)", None)

    run_experiment(
        "InceptionV1",
        CLEAN_DATASET,
        label="InceptionV1 (шум -> чистые)",
        note="двухстадийное обучение",
        config=config,
        model=model,
        progress=progress,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="полный набор экспериментов")
    parser.add_argument("--models", nargs="+", metavar="NAME", help="только эти модели")
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAINING.epochs)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_TRAINING.batch_size)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_TRAINING.num_workers)
    parser.add_argument("--no-progress", action="store_true", help="без прогресс-баров")
    parser.add_argument("--output", type=Path, default=COMPARISON_CSV)
    args = parser.parse_args()

    if not args.all and not args.models:
        parser.error("укажите --all или --models NAME [NAME ...]")

    config = DEFAULT_TRAINING.scaled(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    progress = not args.no_progress

    for name in args.models or ARCHITECTURE_SWEEP:
        run_experiment(
            name,
            CLEAN_DATASET,
            note="очищенные данные",
            config=config,
            progress=progress,
        )

    if args.all:
        run_noise_ablation(config, progress)
        run_two_stage(config, progress)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY.save(args.output)
    REGISTRY.save_json(args.output.with_suffix(".json"))

    print()
    print(REGISTRY.to_frame().to_string())
    print(f"\nсохранено: {args.output}")


if __name__ == "__main__":
    main()
