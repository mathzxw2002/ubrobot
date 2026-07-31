#!/usr/bin/env python3
"""Deterministic OpenAI-compatible planner fixture for Cortex mock tests.

Replaces a real LLM in end-to-end mock validation. It speaks the subset of
the OpenAI API that ``agents.clients.GenericHTTPClient`` uses:

- ``GET /v1/models`` for the connection/checkpoint probe;
- ``POST /v1/chat/completions`` returning either a navigation tool call, a
  final text response after tool execution, or a plain text response.

Decision logic lives in :func:`decide_response` as a pure function so it can
be unit-tested without HTTP or ROS. The server additionally records every
chat request as JSONL so the test harness can assert that UI text reached
the planner unchanged and that the expected tool list was offered.

Configuration is via environment variables only:

- ``FIXTURE_PORT`` (default ``18080``)
- ``FIXTURE_CHECKPOINT`` (default ``mock-planner``)
- ``FIXTURE_TARGET`` (default ``chair``) — navigation goal label
- ``FIXTURE_TIMEOUT_SEC`` (default ``20``) — navigation goal timeout
- ``FIXTURE_LOG`` (default ``/tmp/mock_planner_requests.jsonl``)
- ``FIXTURE_NAV_PATTERN`` — regex; a user prompt matching it triggers the
  navigation tool call (default covers Chinese and English phrasing)
"""

from __future__ import annotations

from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Optional

NAVIGATION_TOOL_NAME = "send_goal_to__ubrobot_navigation_navigate_to_object"
DEFAULT_NAV_PATTERN = r"(走到|走向|导航|navigate|go to|move to|follow)"
EXECUTION_TOOL_CALL_PREFIX = "exec_"


@dataclass(frozen=True)
class FixtureConfig:
    port: int = 18080
    checkpoint: str = "mock-planner"
    target: str = "chair"
    timeout_sec: float = 20.0
    log_path: str = "/tmp/mock_planner_requests.jsonl"
    nav_pattern: str = DEFAULT_NAV_PATTERN


def config_from_env(env: Optional[dict[str, str]] = None) -> FixtureConfig:
    env = os.environ if env is None else env
    return FixtureConfig(
        port=int(env.get("FIXTURE_PORT", "18080")),
        checkpoint=env.get("FIXTURE_CHECKPOINT", "mock-planner"),
        target=env.get("FIXTURE_TARGET", "chair"),
        timeout_sec=float(env.get("FIXTURE_TIMEOUT_SEC", "20")),
        log_path=env.get("FIXTURE_LOG", "/tmp/mock_planner_requests.jsonl"),
        nav_pattern=env.get("FIXTURE_NAV_PATTERN", DEFAULT_NAV_PATTERN),
    )


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user" and message.get("content"):
            return str(message["content"])
    return ""


def _execution_finished(messages: list[dict[str, Any]]) -> bool:
    """True once Cortex fed back an executed tool result (exec_* id)."""
    for message in messages:
        if message.get("role") != "tool":
            continue
        tool_call_id = str(message.get("tool_call_id", ""))
        if tool_call_id.startswith(EXECUTION_TOOL_CALL_PREFIX):
            return True
    return False


def decide_response(
    payload: dict[str, Any], config: FixtureConfig
) -> dict[str, Any]:
    """Map one chat-completions payload to a deterministic assistant message.

    - After an executed tool result is present, return final text (no tools).
    - If the user's task matches the navigation pattern, return exactly one
      ``NavigateToObject`` tool call with the configured target/timeout.
    - Otherwise return a plain text echo; no tools are called.
    """
    messages = list(payload.get("messages") or [])

    if _execution_finished(messages):
        content = (
            f"Navigation to '{config.target}' completed. "
            "The robot executed the semantic navigation capability and stopped."
        )
        return {"role": "assistant", "content": content}

    user_text = _last_user_text(messages)
    if user_text and re.search(config.nav_pattern, user_text, re.IGNORECASE):
        arguments = {"target": config.target, "timeout_sec": config.timeout_sec}
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_fixture_navigate_0",
                    "type": "function",
                    "function": {
                        "name": NAVIGATION_TOOL_NAME,
                        "arguments": json.dumps(arguments),
                    },
                }
            ],
        }

    return {
        "role": "assistant",
        "content": (
            "[No actions needed]. Fixture received: " + (user_text or "<empty>")
        ),
    }


def chat_completion(payload: dict[str, Any], config: FixtureConfig) -> dict[str, Any]:
    message = decide_response(payload, config)
    return {
        "id": "chatcmpl-fixture",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": config.checkpoint,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": (
                    "tool_calls" if message.get("tool_calls") else "stop"
                ),
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def models_listing(config: FixtureConfig) -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": config.checkpoint,
                "object": "model",
                "created": 0,
                "owned_by": "ubrobot-fixture",
            }
        ],
    }


def make_handler(config: FixtureConfig):
    class FixtureHandler(BaseHTTPRequestHandler):
        def _send_json(self, body: dict[str, Any], status: int = 200) -> None:
            encoded = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _record(self, payload: dict[str, Any]) -> None:
            try:
                record = {"ts": time.time(), "payload": payload}
                with open(config.log_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
            except OSError:
                self.log_message("failed to record request to %s", config.log_path)

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                self._send_json(models_listing(config))
            else:
                self._send_json({"error": "not found"}, status=404)

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            if self.path.rstrip("/") != "/v1/chat/completions":
                self._send_json({"error": "not found"}, status=404)
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._send_json({"error": "invalid json"}, status=400)
                return
            self._record(payload)
            self._send_json(chat_completion(payload, config))

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[mock-planner] {format % args}", flush=True)

    return FixtureHandler


def main() -> int:
    config = config_from_env()
    Path(config.log_path).parent.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("0.0.0.0", config.port), make_handler(config))
    print(
        f"[mock-planner] serving checkpoint={config.checkpoint} "
        f"target={config.target} on :{config.port}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
