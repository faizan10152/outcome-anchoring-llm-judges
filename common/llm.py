"""Minimal chat client for local ollama models.

Shared by the agent loop and the judge runs so both use identical decoding
settings; a difference in sampling between the two would be an uncontrolled
variable in the experiment.

Determinism: ollama is asked for temperature 0 and a fixed seed. That makes
runs repeatable in practice on the same machine and model build, but it is not
a cryptographic guarantee - batching and GPU kernels can still reorder
floating-point work. Every trace therefore records the settings used, and the
traces themselves are committed rather than regenerated on demand.
"""
from __future__ import annotations

import json
import time
from typing import Any

import requests


class LLMError(RuntimeError):
    """Raised when the backend cannot be reached or returns an unusable reply."""


class OllamaClient:
    def __init__(self, model: str, base_url: str = "http://localhost:11434",
                 temperature: float = 0.0, seed: int = 7, num_ctx: int = 16384,
                 think: bool | None = None, timeout: int = 600,
                 max_retries: int = 3):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.seed = seed
        self.num_ctx = num_ctx
        self.think = think          # None = leave to the model default
        self.timeout = timeout
        self.max_retries = max_retries

    # --- introspection -------------------------------------------------
    def settings(self) -> dict:
        return {"backend": "ollama", "model": self.model,
                "temperature": self.temperature, "seed": self.seed,
                "num_ctx": self.num_ctx, "think": self.think}

    def available_models(self) -> list[str]:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=30)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except requests.RequestException as exc:
            raise LLMError(f"cannot reach ollama at {self.base_url}: {exc}") from exc

    # --- the one call that matters --------------------------------------
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Send a chat turn and return the assistant message dict.

        Returns the raw message: {"role", "content", optionally "thinking"
        and "tool_calls"}. Transport errors are retried; a reply that parses
        is returned as-is, including an empty one, so the caller can decide
        what an empty turn means.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature, "seed": self.seed,
                        "num_ctx": self.num_ctx},
        }
        if tools:
            payload["tools"] = tools
        if self.think is not None:
            payload["think"] = self.think

        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = requests.post(f"{self.base_url}/api/chat", json=payload,
                                  timeout=self.timeout)
                r.raise_for_status()
                body = r.json()
            except requests.HTTPError as exc:
                # Models without a reasoning channel reject "think" outright.
                # Drop it once and carry on rather than failing a whole run:
                # the parameter is a no-op for them anyway.
                if "think" in payload and _is_think_unsupported(exc):
                    payload.pop("think")
                    self.think = None
                    continue
                last_exc = exc
                if attempt == self.max_retries:
                    break
                time.sleep(2 * attempt)
                continue
            except (requests.RequestException, json.JSONDecodeError) as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    break
                time.sleep(2 * attempt)
                continue

            if "message" not in body:
                last_exc = LLMError(f"reply without a message field: {body!r}")
                if attempt == self.max_retries:
                    break
                time.sleep(2 * attempt)
                continue
            return body["message"]

        raise LLMError(f"{self.model}: chat failed after {self.max_retries} "
                       f"attempts: {last_exc}")


def _is_think_unsupported(exc: requests.HTTPError) -> bool:
    """True if ollama rejected the request because the model cannot think."""
    try:
        text = (exc.response.text or "").lower()
    except Exception:
        return False
    return "think" in text and ("support" in text or "capab" in text)


def normalise_tool_calls(message: dict) -> list[dict]:
    """Return tool calls in a single shape, whatever the backend emitted.

    Handles the two variations seen across local models: a missing call id,
    and arguments delivered as a JSON string rather than an object.
    """
    calls = []
    for i, tc in enumerate(message.get("tool_calls") or []):
        fn = tc.get("function", {})
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {"__unparsed__": args}
        if not isinstance(args, dict):
            args = {"__unparsed__": args}
        calls.append({
            "id": tc.get("id") or f"call_{i}",
            "type": "function",
            "function": {"name": fn.get("name", ""), "arguments": args},
        })
    return calls
