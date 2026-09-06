"""Инференс обученной модели по папке изображений.

    python scripts/predict.py --model InceptionV1 \
        --checkpoint checkpoints/inceptionv1.pth \
        --images data/processed/clean_test_images \
        --output results/submission.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deepfake.config import (  # noqa: E402
    CLEAN_TEST_IMAGES,
    DEFAULT_TRAINING,
    RESULTS_DIR,
    get_device,
    use_amp,
)
from deepfake.data import ImageFolderFlat  # noqa: E402
from deepfake.experiment import transforms_for  # noqa: E402
from deepfake.models import build_model  # noqa: E402


@torch.no_grad()
def predict_folder(
    model,
    images_dir: Path,
    transform,
    threshold: float,
    batch_size: int,
    num_workers: int,
) -> list[tuple[int, int]]:
    device = get_device()
    model.to(device).eval()

    dataset = ImageFolderFlat(images_dir, transform)
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )

    rows: list[tuple[int, int]] = []
    for inputs, ids in loader:
        inputs = inputs.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp(device)):
            logits = model(inputs)
        probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
        predictions = (probs >= threshold).astype(int).tolist()
        rows.extend(zip(ids.tolist(), predictions, strict=True))

    rows.sort()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="имя архитектуры из реестра")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, default=CLEAN_TEST_IMAGES)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "submission.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_TRAINING.num_workers)
    args = parser.parse_args()

    if not args.images.exists():
        raise SystemExit(f"папка с изображениями не найдена: {args.images}")
    if not args.checkpoint.exists():
        raise SystemExit(f"чекпоинт не найден: {args.checkpoint}")

    model = build_model(args.model)
    model.load_state_dict(torch.load(args.checkpoint, weights_only=True))

    _, eval_transform = transforms_for(args.model, DEFAULT_TRAINING.image_size)
    rows = predict_folder(
        model,
        args.images,
        eval_transform,
        args.threshold,
        args.batch_size,
        args.num_workers,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "target_feature"])
        writer.writerows(rows)

    fake = sum(label for _, label in rows)
    print(f"{args.output}: {len(rows)} строк, fake {fake} ({fake / max(len(rows), 1):.1%})")


if __name__ == "__main__":
    main()
