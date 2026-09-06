"""Собирает RESULTS.md и график сравнения из results/model_comparison.csv.

Отчёт генерируется из данных, а не пишется руками — цифры в нём не могут
разъехаться с тем, что реально посчитали.

    python scripts/report.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from deepfake.config import COMPARISON_CSV, FIGURES_DIR, PROJECT_ROOT  # noqa: E402
from deepfake.metrics import ExperimentRegistry  # noqa: E402

TEMPLATE = """# Результаты

Все строки получены по единому протоколу: одинаковый split, одинаковый бюджет
обучения, свежие веса на каждый эксперимент, чекпоинт и порог отобраны по
валидации, метрики сняты с отложенной тестовой выборки.

Файл сгенерирован скриптом `scripts/report.py` из `results/model_comparison.csv` —
цифры здесь не могут разойтись с посчитанными.

## Сравнение моделей

{table}

![Сравнение моделей](results/figures/model_comparison.png)

## Как читать таблицу

**F1 — основная метрика.** Классы несбалансированы (83% реальных лиц), поэтому
accuracy вводит в заблуждение: модель, всегда отвечающая «real», получила бы
около 0.83 без всякой пользы.

**ROC-AUC считается по вероятностям** и не зависит от выбранного порога — он
показывает качество ранжирования отдельно от качества калибровки. AUC от жёстких
меток, который часто считают по ошибке, — это балансированная точность, другая
величина.

**Порог подобран по валидации**, а не зафиксирован на 0.5. При дисбалансе 5:1
оптимум по F1 обычно лежит заметно ниже, и `argmax` (эквивалент порога 0.5)
систематически недобирает recall.

## Что показывают сравнения

{ablations}

## Воспроизведение

```bash
python scripts/prepare_data.py --all
python scripts/train.py --all
python scripts/report.py
```
"""

ABLATION_TEMPLATE = """### {title}

{body}
"""


def build_ablations(frame) -> str:
    """Формулирует выводы из пар экспериментов, если обе строки есть в таблице."""
    blocks = []

    def compare(title, left, right, question):
        if left not in frame.index or right not in frame.index:
            return
        delta = frame.loc[left, "f1"] - frame.loc[right, "f1"]
        direction = "выше" if delta > 0 else "ниже"
        blocks.append(
            ABLATION_TEMPLATE.format(
                title=title,
                body=(
                    f"{question}\n\n"
                    f"- `{left}` — F1 **{frame.loc[left, 'f1']:.4f}**\n"
                    f"- `{right}` — F1 **{frame.loc[right, 'f1']:.4f}**\n\n"
                    f"Разница: **{abs(delta):.4f}** F1 ({direction} у `{left}`)."
                ),
            )
        )

    compare(
        "Вклад очистки от шума",
        "InceptionV1",
        "InceptionV1 (шумные)",
        "Датасеты отличаются только предобработкой — те же изображения, то же "
        "разбиение. Значит, разница целиком относится на счёт удаления шума.",
    )
    compare(
        "Вклад частотной ветви",
        "InceptionV1+Freq",
        "InceptionV1",
        "Модели отличаются только тем, что во второй на вход дополнительно "
        "подаётся высокочастотный остаток `x - среднее по окну`.",
    )
    compare(
        "Цена обучения с нуля",
        "ResNet18 (ImageNet)",
        "ResNet18",
        "Одна и та же архитектура, один и тот же бюджет обучения; отличается "
        "только стартовая инициализация — ImageNet против случайной.",
    )
    compare(
        "Шум как предобучение",
        "InceptionV1 (шум -> чистые)",
        "InceptionV1",
        "Двухстадийная схема против обучения сразу на очищенных данных.",
    )

    return "\n".join(blocks) if blocks else "_Недостаточно экспериментов для сравнений._"


def plot_comparison(frame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = frame.sort_values("f1")

    fig, axis = plt.subplots(figsize=(9, 0.45 * len(ordered) + 2))
    axis.barh(ordered.index, ordered["f1"], color="#4C72B0")
    axis.set_xlabel("F1 на тестовой выборке")
    axis.set_xlim(max(0.0, ordered["f1"].min() - 0.05), 1.0)
    axis.grid(axis="x", alpha=0.3)
    axis.set_title("Сравнение архитектур")

    for index, value in enumerate(ordered["f1"]):
        axis.text(value, index, f" {value:.4f}", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=COMPARISON_CSV)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "RESULTS.md")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(
            f"нет файла с результатами: {args.input}\n"
            "Сначала запустите: python scripts/train.py --all"
        )

    registry = ExperimentRegistry.load(args.input)
    frame = registry.to_frame()
    if frame.empty:
        raise SystemExit(f"{args.input} пустой")

    columns = ["f1", "roc_auc", "accuracy", "precision", "recall", "threshold", "dataset", "epochs"]
    columns = [c for c in columns if c in frame.columns]

    figure_path = FIGURES_DIR / "model_comparison.png"
    plot_comparison(frame, figure_path)

    args.output.write_text(
        TEMPLATE.format(
            table=frame[columns].to_markdown(),
            ablations=build_ablations(frame),
        ),
        encoding="utf-8",
    )

    print(f"{args.output}: {len(frame)} строк")
    print(f"{figure_path}")


if __name__ == "__main__":
    main()
