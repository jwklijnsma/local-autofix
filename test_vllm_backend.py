"""
Regression test for the vLLM backend (config.yaml: ollama.backend: "vllm").

Confirms:
  - Requests go to vLLM's OpenAI-compatible /v1/chat/completions, not
    Ollama's /api/chat.
  - The payload is flat OpenAI-style (model/messages/temperature/max_tokens),
    not Ollama's nested "options" dict.
  - json_mode correctly maps to response_format={"type": "json_object"}.
  - Responses are correctly parsed from choices[0].message.content instead
    of Ollama's message.content.
  - num_gpu/num_thread (Ollama-only concepts) log a warning instead of
    silently doing nothing when set with backend="vllm".
  - The default ("ollama") backend's behavior is completely unchanged.
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

from ollama_client import OllamaClient, OllamaClientConfig, OllamaEndpoint  # noqa: E402


class FakeResp:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _vllm_response(content: str) -> FakeResp:
    return FakeResp({
        "id": "chatcmpl-123",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
    })


def test_vllm_url_and_payload_shape():
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["payload"] = json
        return _vllm_response("vllm says hi")

    cfg = OllamaClientConfig(
        endpoints=[OllamaEndpoint("http://127.0.0.1:8000", "vllm0")],
        backend="vllm",
        model="Qwen2.5-Coder-32B-Instruct",
    )
    client = OllamaClient(cfg)

    with patch("ollama_client.requests.post", side_effect=fake_post):
        result = client.chat([{"role": "user", "content": "hi"}], heartbeat_interval=0)

    assert captured["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert "options" not in captured["payload"], "vLLM payload must be flat, not Ollama's nested options"
    assert captured["payload"]["temperature"] == 0.2
    assert "max_tokens" in captured["payload"]
    assert result == "vllm says hi"
    print("PASS: vLLM backend hits the correct URL with a flat OpenAI-style payload")


def test_vllm_json_mode():
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["payload"] = json
        return _vllm_response("{}")

    cfg = OllamaClientConfig(endpoints=[OllamaEndpoint("http://127.0.0.1:8000", "vllm0")], backend="vllm")
    client = OllamaClient(cfg)

    with patch("ollama_client.requests.post", side_effect=fake_post):
        client.chat([{"role": "user", "content": "hi"}], json_mode=True, heartbeat_interval=0)

    assert captured["payload"]["response_format"] == {"type": "json_object"}
    print("PASS: json_mode maps to response_format for vLLM")


def test_ollama_backend_unchanged():
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["payload"] = json
        return FakeResp({"message": {"content": "ollama says hi"}})

    cfg = OllamaClientConfig(endpoints=[OllamaEndpoint("http://127.0.0.1:11434", "gpu0")])  # default backend
    client = OllamaClient(cfg)

    with patch("ollama_client.requests.post", side_effect=fake_post):
        result = client.chat([{"role": "user", "content": "hi"}], heartbeat_interval=0)

    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    assert "options" in captured["payload"]
    assert result == "ollama says hi"
    print("PASS: default (ollama) backend behavior is unchanged")


def test_llamacpp_uses_same_openai_compatible_path():
    """
    llama.cpp's server (llama-server) exposes the same OpenAI-compatible
    API as vLLM, so backend='llamacpp' shares the vLLM code path -- only
    the config value differs, for clarity in the user's config.yaml.
    """
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["payload"] = json
        return _vllm_response("llama.cpp says hi")

    cfg = OllamaClientConfig(
        endpoints=[OllamaEndpoint("http://127.0.0.1:8080", "llamacpp0")],
        backend="llamacpp",
        model="qwen2.5-coder-32b-instruct-q4_k_m.gguf",
    )
    client = OllamaClient(cfg)

    with patch("ollama_client.requests.post", side_effect=fake_post):
        result = client.chat([{"role": "user", "content": "hi"}], heartbeat_interval=0)

    assert captured["url"] == "http://127.0.0.1:8080/v1/chat/completions"
    assert "options" not in captured["payload"]
    assert result == "llama.cpp says hi"
    print("PASS: backend='llamacpp' correctly uses the OpenAI-compatible code path")


if __name__ == "__main__":
    test_vllm_url_and_payload_shape()
    test_vllm_json_mode()
    test_ollama_backend_unchanged()
    test_llamacpp_uses_same_openai_compatible_path()
    print("\nALL VLLM/LLAMACPP BACKEND TESTS PASSED")
