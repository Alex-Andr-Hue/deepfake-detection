"""Архитектуры: форма выхода, детерминизм инициализации, частотная ветвь."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from deepfake.models import (
    MODEL_FACTORIES,
    FrequencyAware,
    HighPassResidual,
    InceptionV1,
    build_model,
    count_parameters,
    mean_log_spectrum,
)


@pytest.mark.parametrize("name", sorted(MODEL_FACTORIES))
def test_every_model_accepts_expected_input(name):
    """Каждая модель из реестра принимает 256x256 и выдаёт два логита."""
    model = build_model(name).eval()
    with torch.no_grad():
        output = model(torch.rand(2, 3, 256, 256))

    assert output.shape == (2, 2)
    assert torch.isfinite(output).all()


def test_unknown_model_raises():
    with pytest.raises(KeyError, match="неизвестная модель"):
        build_model("NoSuchModel")


def test_build_model_is_deterministic():
    """Фабрика с фиксированным seed даёт одинаковые начальные веса."""
    first = build_model("CNN").state_dict()
    second = build_model("CNN").state_dict()

    for key, value in first.items():
        assert torch.equal(value, second[key]), key


def test_build_model_returns_fresh_instance():
    """Каждый вызов — новый объект, а не общая глобальная модель."""
    first = build_model("CNN")
    second = build_model("CNN")

    with torch.no_grad():
        first[-1].bias.fill_(3.0)

    assert not torch.equal(first[-1].bias, second[-1].bias)


def test_highpass_removes_constant_component():
    """Равномерное изображение целиком лежит в низких частотах."""
    residual = HighPassResidual()(torch.full((2, 3, 16, 16), 0.5))
    assert residual.abs().max().item() < 1e-6


def test_highpass_keeps_texture():
    residual = HighPassResidual()(torch.rand(2, 3, 16, 16))
    assert residual.abs().max().item() > 0.01


def test_highpass_weights_are_not_trainable():
    """Фильтр фиксированный — это предобработка, а не обучаемый слой."""
    highpass = HighPassResidual()
    assert count_parameters(highpass) == 0
    assert "weight" in dict(highpass.named_buffers())


def test_highpass_rejects_even_kernel():
    with pytest.raises(ValueError, match="нечётным"):
        HighPassResidual(kernel=2)


def test_frequency_aware_doubles_input_channels():
    """Обёртка подаёт в backbone 6 каналов: RGB плюс высокочастотный остаток."""
    model = FrequencyAware(InceptionV1(6, 2)).eval()
    with torch.no_grad():
        output = model(torch.rand(2, 3, 64, 64))

    assert output.shape == (2, 2)


def test_frequency_branch_adds_few_parameters():
    """Растёт только первая свёртка, а не вся сеть."""
    plain = count_parameters(InceptionV1(3, 2))
    aware = count_parameters(FrequencyAware(InceptionV1(6, 2)))

    assert aware > plain
    assert (aware - plain) / plain < 0.01


def test_mean_log_spectrum_shape_and_symmetry():
    rng = np.random.default_rng(0)
    spectrum = mean_log_spectrum(rng.random((5, 32, 32)))

    assert spectrum.shape == (32, 32)
    # у вещественного сигнала нулевая частота после fftshift — максимум спектра
    assert np.unravel_index(spectrum.argmax(), spectrum.shape) == (16, 16)


def test_mean_log_spectrum_rejects_wrong_shape():
    with pytest.raises(ValueError, match=r"\(N, H, W\)"):
        mean_log_spectrum(np.zeros((32, 32)))
