"""
Offline test for the circuit breaker: confirms that a consistently-failing
endpoint gets skipped after circuit_breaker_failure_threshold consecutive
failures, and gets automatically retried after circuit_breaker_cooldown_seconds.

This directly addresses the real-world case where one Ollama server is
broken/misconfigured while others work fine -- without the breaker, every
other request round-robins onto the broken one and pays a full
request_timeout before falling back.
"""
import sys
import time
from pathlib import Path
from unittest.mock import patch

import requests

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


def test_breaker_routes_around_broken_endpoint():
    hits = {"gpu0": 0, "gpu1": 0}

    def fake_post(url, json=None, timeout=None):
        if "10.0.3.24" in url:
            hits["gpu1"] += 1
            raise requests.exceptions.ConnectionError("gpu1 is broken")
        hits["gpu0"] += 1
        return FakeResp({"message": {"content": "ok"}})

    cfg = OllamaClientConfig(
        endpoints=[
            OllamaEndpoint("http://10.0.3.140:11434", "gpu0"),
            OllamaEndpoint("http://10.0.3.24:11434", "gpu1"),
        ],
        max_retries=1,
        circuit_breaker_failure_threshold=2,
        circuit_breaker_cooldown_seconds=120,
    )
    client = OllamaClient(cfg)

    with patch("ollama_client.requests.post", side_effect=fake_post):
        for i in range(10):
            try:
                client.chat([{"role": "user", "content": f"file {i}"}], heartbeat_interval=0)
            except RuntimeError:
                pass

    assert hits["gpu1"] <= 3, f"expected the breaker to sharply limit gpu1 hits, got {hits['gpu1']}"
    assert hits["gpu0"] >= 7, f"expected most traffic on healthy gpu0, got {hits['gpu0']}"
    print(f"PASS: broken endpoint hit only {hits['gpu1']}/10 times (breaker engaged), gpu0 got {hits['gpu0']}/10")


def test_breaker_recovers_after_cooldown():
    state = {"gpu1_broken": True}

    def fake_post(url, json=None, timeout=None):
        if "10.0.3.24" in url and state["gpu1_broken"]:
            raise requests.exceptions.ConnectionError("gpu1 down")
        return FakeResp({"message": {"content": "ok"}})

    cfg = OllamaClientConfig(
        endpoints=[
            OllamaEndpoint("http://10.0.3.140:11434", "gpu0"),
            OllamaEndpoint("http://10.0.3.24:11434", "gpu1"),
        ],
        max_retries=1,
        circuit_breaker_failure_threshold=2,
        circuit_breaker_cooldown_seconds=1,  # short, just for test speed
    )
    client = OllamaClient(cfg)

    with patch("ollama_client.requests.post", side_effect=fake_post):
        for i in range(4):
            try:
                client.chat([{"role": "user", "content": "x"}], heartbeat_interval=0)
            except RuntimeError:
                pass

        state["gpu1_broken"] = False
        time.sleep(1.2)  # let the cooldown expire

        hits_after = {"gpu1": 0}

        def counting_post(url, **kw):
            if "10.0.3.24" in url:
                hits_after["gpu1"] += 1
            return fake_post(url, **kw)

        with patch("ollama_client.requests.post", side_effect=counting_post):
            for i in range(4):
                client.chat([{"role": "user", "content": "y"}], heartbeat_interval=0)

    assert hits_after["gpu1"] > 0, "endpoint should be retried again after cooldown expires"
    print(f"PASS: recovered endpoint used again after cooldown ({hits_after['gpu1']} hits post-recovery)")


if __name__ == "__main__":
    test_breaker_routes_around_broken_endpoint()
    test_breaker_recovers_after_cooldown()
    print("\nALL CIRCUIT BREAKER TESTS PASSED")
