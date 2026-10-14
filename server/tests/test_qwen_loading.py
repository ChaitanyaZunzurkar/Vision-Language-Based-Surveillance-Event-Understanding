import sys
from types import ModuleType, SimpleNamespace

import pytest

from server.src.event_understanding import qwen
from server.src.event_understanding.qwen import VLMEventUnderstanding


def _chat_evidence():
    return [{
        "event_id": "event-1",
        "video_id": "video-1",
        "event_type": "walking",
        "start_sec": 1.0,
        "end_sec": 2.0,
        "description": "A person walks across the hallway.",
        "confidence": 0.9,
        "anomaly_category": "Normal",
        "anomaly_confidence": 0.98,
    }]


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


def test_groq_chat_uses_configured_model_without_loading_qwen(monkeypatch):
    monkeypatch.setenv("CHAT_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")
    vlm = VLMEventUnderstanding()
    calls = []

    def groq_answer(question, facts, history):
        calls.append((question, facts, history))
        return "A person walks across the hallway between 1 and 2 seconds."

    monkeypatch.setattr(vlm, "_generate_groq_answer", groq_answer)
    monkeypatch.setattr(
        vlm,
        "_load_model",
        lambda: pytest.fail("Successful Groq responses must not load Qwen"),
    )

    answer = vlm.generate_grounded_answer(
        "What happened?", _chat_evidence(), [{"role": "user", "content": "Earlier"}]
    )

    assert answer == "A person walks across the hallway between 1 and 2 seconds."
    assert calls[0][0] == "What happened?"
    assert calls[0][1][0]["event_id"] == "event-1"
    assert calls[0][2] == [{"role": "user", "content": "Earlier"}]


def test_groq_client_uses_env_model_and_grounded_event_context(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")
    calls = {}
    groq = ModuleType("groq")

    class Client:
        def __init__(self, **options):
            calls["client_options"] = options
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create_completion)
            )

        @staticmethod
        def create_completion(**options):
            calls["completion_options"] = options
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="A person walks across the hallway from 1 to 2 seconds."
                        )
                    )
                ]
            )

    groq.Groq = Client
    monkeypatch.setitem(sys.modules, "groq", groq)

    answer = VLMEventUnderstanding._generate_groq_answer(
        "What happened?",
        _chat_evidence(),
        [{"role": "user", "content": "Earlier"}],
    )

    assert "person walks" in answer
    assert calls["client_options"] == {
        "api_key": "test-key",
        "max_retries": 0,
        "timeout": 12.0,
    }
    assert calls["completion_options"]["model"] == "openai/gpt-oss-20b"
    assert calls["completion_options"]["messages"][1] == {
        "role": "user",
        "content": "Earlier",
    }
    assert "Verified event records" in calls["completion_options"]["messages"][2]["content"]


def test_groq_chat_failure_falls_back_to_qwen(monkeypatch):
    monkeypatch.setenv("CHAT_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    vlm = VLMEventUnderstanding()
    monkeypatch.setattr(
        vlm,
        "_generate_groq_answer",
        lambda *args: (_ for _ in ()).throw(RuntimeError("API unavailable")),
    )
    monkeypatch.setattr(
        vlm,
        "_generate_qwen_answer",
        lambda messages, evidence_ids: "Qwen fallback answer based on the event records.",
    )

    answer = vlm.generate_grounded_answer(
        "What happened?", _chat_evidence(), []
    )

    assert answer == "Qwen fallback answer based on the event records."
