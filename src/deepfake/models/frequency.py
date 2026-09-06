"""Частотная ветвь — приём, специфичный именно для детекции дипфейков.

Генеративные модели оставляют регулярные высокочастотные следы: сетку от
транспонированных свёрток, характерные пики в спектре, неестественную
статистику соседних пикселей. В RGB эти следы слабы по амплитуде и тонут в
контенте — лице, фоне, освещении.

``HighPassResidual`` вытаскивает их явно, ``FrequencyAware`` подаёт в backbone
исходное изображение и остаток одновременно, так что сети не нужно заново
изобретать высокочастотный фильтр первыми свёртками.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HighPassResidual(nn.Module):
    """``x - среднее по окну``: оставляет только высокие частоты.

    Фильтр фиксированный (буфер, а не параметр) — он не обучается и не
    участвует в оптимизации, это детерминированная предобработка внутри графа.
    """

    def __init__(self, kernel: int = 3, channels: int = 3):
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError(f"размер окна должен быть нечётным, получено {kernel}")

        self.pad = kernel // 2
        self.channels = channels
        weight = torch.ones(channels, 1, kernel, kernel) / (kernel * kernel)
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded = F.pad(x, (self.pad,) * 4, mode="reflect")
        blurred = F.conv2d(padded, self.weight, groups=self.channels)
        return x - blurred


class FrequencyAware(nn.Module):
    """Оборачивает backbone, удваивая число входных каналов.

    На вход backbone идёт ``cat([x, highpass(x)])``, поэтому backbone должен
    быть создан с шестью входными каналами вместо трёх.
    """

    def __init__(self, backbone: nn.Module, kernel: int = 3):
        super().__init__()
        self.highpass = HighPassResidual(kernel)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(torch.cat([x, self.highpass(x)], dim=1))


def mean_log_spectrum(images: np.ndarray) -> np.ndarray:
    """Средний лог-амплитудный спектр Фурье по набору изображений.

    Диагностика, а не часть модели: разность средних спектров real и fake
    показывает, есть ли частотный признак в данных, ещё до всякого обучения.

    Args:
        images: массив (N, H, W) в градациях серого, значения 0..1.

    Returns:
        Спектр (H, W) с нулевой частотой в центре.
    """
    if images.ndim != 3:
        raise ValueError(f"ожидается (N, H, W), получено {images.shape}")

    spectra = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(images), axes=(-2, -1))))
    return spectra.mean(axis=0)
