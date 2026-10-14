import sys
from types import ModuleType, SimpleNamespace

import pytest

from server.src.event_understanding import qwen
from server.src.event_understanding.qwen import VLMEventUnderstanding


def _install_model_modules(monkeypatch, model_class):
    torch = ModuleType("torch")
    torch.float16 = "float16"
    torch.float32 = "float32"

    transformers = ModuleType("transformers")
    transformers.AutoProcessor = SimpleNamespace(
        from_pretrained=lambda model_name: object()
    )
    transformers.Qwen2_5_VLForConditionalGeneration = model_class
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return transformers


def test_cuda_on_windows_skips_bitsandbytes_and_loads_fp16(monkeypatch):
    calls = []

    class Model:
        @staticmethod
        def from_pretrained(model_name, **options):
            calls.append(options)
            return SimpleNamespace(eval=lambda: None)

    transformers = _install_model_modules(monkeypatch, Model)
    monkeypatch.setattr(qwen.sys, "platform", "win32")
    vlm = VLMEventUnderstanding(device="cuda:0")

    vlm._load_model_unlocked()

    assert len(calls) == 1
    assert calls[0]["device_map"] == "auto"
    assert calls[0]["torch_dtype"] == "float16"
    assert "quantization_config" not in calls[0]
    assert vlm.model_available
    assert not hasattr(transformers, "BitsAndBytesConfig")


def test_bitsandbytes_initialization_error_retries_with_fp16(monkeypatch):
    calls = []

    class Model:
        @staticmethod
        def from_pretrained(model_name, **options):
            calls.append(options.copy())
            if len(calls) == 1:
                raise ImportError(
                    "Using bitsandbytes 4-bit quantization requires bitsandbytes"
                )
            return SimpleNamespace(eval=lambda: None)

    transformers = _install_model_modules(monkeypatch, Model)
    transformers.BitsAndBytesConfig = lambda **options: options
    monkeypatch.setattr(qwen.sys, "platform", "linux")
    monkeypatch.setattr(
        VLMEventUnderstanding, "_supports_4bit_quantization", staticmethod(lambda: True)
    )
    vlm = VLMEventUnderstanding(device="cuda:0")

    vlm._load_model_unlocked()

    assert len(calls) == 2
    assert calls[0]["quantization_config"]["load_in_4bit"] is True
    assert "quantization_config" not in calls[1]
    assert calls[1]["torch_dtype"] == "float16"
    assert vlm.model_available


def test_non_bitsandbytes_model_loading_errors_are_not_hidden(monkeypatch):
    class Model:
        @staticmethod
        def from_pretrained(model_name, **options):
            raise RuntimeError("CUDA out of memory")

    _install_model_modules(monkeypatch, Model)
    monkeypatch.setattr(qwen.sys, "platform", "win32")
    vlm = VLMEventUnderstanding(device="cuda:0")

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        vlm._load_model_unlocked()
