"""Реестр архитектур.

Каждая модель создаётся фабрикой, а не берётся из глобальной переменной. Это
гарантирует, что эксперимент стартует со свежих весов и его результат не
смешивается с результатом предыдущего запуска — самая частая причина, по
которой сравнение моделей оказывается недействительным.
"""

from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn

from ..config import SEED, set_seed
from .cnn import make_cnn
from .frequency import FrequencyAware, HighPassResidual, mean_log_spectrum
from .inception import InceptionV1, InceptionV3
from .resnet import ResNet18, ResNet50
from .unet import UNetForClassification

__all__ = [
    "MODEL_FACTORIES",
    "build_model",
    "count_parameters",
    "FrequencyAware",
    "HighPassResidual",
    "InceptionV1",
    "InceptionV3",
    "ResNet18",
    "ResNet50",
    "UNetForClassification",
    "make_cnn",
    "mean_log_spectrum",
    "make_pretrained_resnet18",
]


def make_pretrained_resnet18(num_classes: int = 2) -> nn.Module:
    """ResNet18 с весами ImageNet и заменённой головой.

    Точка отсчёта, показывающая цену обучения с нуля: если написанные руками
    сети проигрывают дообученному за несколько эпох ResNet18 — это
    содержательный результат.

    Требует нормировки ImageNet: с чужими константами предобученные веса
    работают заметно хуже.
    """
    from torchvision.models import ResNet18_Weights, resnet18

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


MODEL_FACTORIES: dict[str, Callable[[], nn.Module]] = {
    "CNN": make_cnn,
    "ResNet18": ResNet18,
    "ResNet50": ResNet50,
    "UNet": UNetForClassification,
    "InceptionV1": lambda: InceptionV1(3, 2),
    "InceptionV3": lambda: InceptionV3(3, 2),
    "InceptionV1+Freq": lambda: FrequencyAware(InceptionV1(6, 2)),
    "ResNet18 (ImageNet)": make_pretrained_resnet18,
}

#: модели, которым нужна нормировка ImageNet вместо статистик датасета
PRETRAINED_MODELS = frozenset({"ResNet18 (ImageNet)"})


def build_model(name: str, seed: int = SEED) -> nn.Module:
    """Создаёт свежую модель по имени, зафиксировав инициализацию."""
    if name not in MODEL_FACTORIES:
        raise KeyError(
            f"неизвестная модель {name!r}; доступны: {sorted(MODEL_FACTORIES)}"
        )
    set_seed(seed)
    return MODEL_FACTORIES[name]()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
