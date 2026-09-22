"""Запускает эксперименты по единому протоколу и пишет итоговую таблицу.

    python scripts/train.py --all                  # весь набор сравнений
    python scripts/train.py --models ResNet18      # только одна модель
    python scripts/train.py --all --epochs 5       # быстрый прогон
    python scripts/train.py --all --resume --limit 2   # следующие две модели

Прогон рассчитан на среды с ограничением по времени: таблица дописывается
после каждого эксперимента, а ``--resume`` подхватывает уже посчитанное.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deepfake.config import (  # noqa: E402
    CHECKPOINTS_DIR,
    CLEAN_DATASET,
    COMPARISON_CSV,
    DEFAULT_TRAINING,
    NOISY_DATASET,
    RESULTS_DIR,
)
from deepfake.experiment import run_experiment  # noqa: E402
from deepfake.metrics import REGISTRY, ExperimentRegistry  # noqa: E402
from deepfake.models import MODEL_FACTORIES  # noqa: E402

#: базовое сравнение архитектур: все на очищенных данных, всё остальное одинаково
ARCHITECTURE_SWEEP = list(MODEL_FACTORIES)

ABLATION_NOISY = "InceptionV1 (шумные)"
ABLATION_TWO_STAGE = "InceptionV1 (шум -> чистые)"


def save_registry(output: Path) -> None:
    """Перезаписывает таблицу целиком после каждого эксперимента.

    Сохранение только в конце прогона означает, что обрыв сессии по таймауту
    стирает результаты всех досчитанных моделей вместе с недосчитанной.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.save(output)
    REGISTRY.save_json(output.with_suffix(".json"))


def run_architecture(name: str, config, checkpoints: Path, progress: bool) -> None:
    run_experiment(
        name,
        CLEAN_DATASET,
        note="очищенные данные",
        config=config,
        checkpoints_dir=checkpoints,
        progress=progress,
    )


def run_noise_ablation(config, checkpoints: Path, progress: bool) -> None:
    """Вклад очистки: та же модель на шумной версии того же разбиения."""
    run_experiment(
        "InceptionV1",
        NOISY_DATASET,
        label=ABLATION_NOISY,
        note="без удаления шума",
        config=config,
        checkpoints_dir=checkpoints,
        progress=progress,
    )


def run_two_stage(config, checkpoints: Path, progress: bool) -> None:
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
        checkpoints_dir=checkpoints,
        progress=progress,
    )
    REGISTRY.results.pop("InceptionV1 (стадия 1: шум)", None)

    run_experiment(
        "InceptionV1",
        CLEAN_DATASET,
        label=ABLATION_TWO_STAGE,
        note="двухстадийное обучение",
        config=config,
        checkpoints_dir=checkpoints,
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
    parser.add_argument(
        "--checkpoints",
        type=Path,
        default=CHECKPOINTS_DIR,
        help="куда складывать веса; вынесите наружу репозитория, если он пересоздаётся",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="подхватить таблицу из --output и пропустить посчитанные эксперименты",
    )
    parser.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="остановиться после N экспериментов — для сессий с лимитом по времени",
    )
    args = parser.parse_args()

    if not args.all and not args.models:
        parser.error("укажите --all или --models NAME [NAME ...]")

    config = DEFAULT_TRAINING.scaled(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    progress = not args.no_progress

    if args.resume:
        REGISTRY.results.update(ExperimentRegistry.load(args.output).results)
        print(f"возобновление: в таблице уже {len(REGISTRY.results)} эксперим.")

    plan: list[tuple[str, Callable[[], None]]] = [
        (name, partial(run_architecture, name, config, args.checkpoints, progress))
        for name in args.models or ARCHITECTURE_SWEEP
    ]
    if args.all:
        noisy = partial(run_noise_ablation, config, args.checkpoints, progress)
        two_stage = partial(run_two_stage, config, args.checkpoints, progress)
        plan.append((ABLATION_NOISY, noisy))
        plan.append((ABLATION_TWO_STAGE, two_stage))

    pending = [(label, run) for label, run in plan if label not in REGISTRY.results]
    scheduled = pending[: args.limit] if args.limit else pending

    for _label, run in scheduled:
        run()
        save_registry(args.output)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_registry(args.output)

    print()
    print(REGISTRY.to_frame().to_string())
    print(f"\nсохранено: {args.output}")

    remaining = [label for label, _ in pending[len(scheduled) :]]
    if remaining:
        print(f"осталось {len(remaining)}: {', '.join(remaining)}")
        print("продолжить: python scripts/train.py --all --resume --limit 2")


if __name__ == "__main__":
    main()
