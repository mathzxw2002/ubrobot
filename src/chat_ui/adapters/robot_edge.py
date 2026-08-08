"""Robot Edge backend adapter for TaskRuntime.

Connects the Operator Console's TaskRuntime to a remote Robot Edge service over
HTTP. The adapter owns no hardware knowledge and never fabricates live data: if
the Edge is unreachable or no token is configured, execution fails clearly
rather than producing mock feedback.

Implements the TaskBackend protocol used by TaskRuntime:

    execute(task, *, on_feedback) -> str
    cancel_active() -> bool
    emergency_stop() -> bool
    close() -> None
"""

from __future__ import annotations

import json
import logging
import os
import threading

logger = logging.getLogger("ubrobot.operator_console.robot_edge_backend")
import time
from datetime import datetime, timezone
from typing import Callable
from uuid import uuid4

import httpx


class RobotEdgeBackend:
    """TaskBackend that drives a remote Robot Edge service."""

    def __init__(
        self,
        edge_url: str,
        operator_id: str,
        token_file: str | None = None,
        token: str | None = None,
    ) -> None:
        if not edge_url:
            raise ValueError("edge_url is required")
        if not operator_id:
            raise ValueError("operator_id is required")

        self._edge_url = edge_url.rstrip("/")
        self._operator_id = operator_id
        self._token = self._load_token(token_file, token)
        # No default control token is allowed: fixture and real runs must both
        # inject a token. Without one the backend cannot safely authenticate.
        if not self._token:
            raise RuntimeError(
                "Robot Edge token not configured. Set UBROBOT_EDGE_TOKEN or "
                "UBROBOT_EDGE_TOKEN_FILE to a server-side token mapping. "
                "No default control token is permitted."
            )

        self._client = httpx.Client(
            base_url=self._edge_url,
            timeout=30.0,
            headers={"Authorization": f"Bearer {self._token}"},
            # The Edge is a direct local/direct-network peer; never route the
            # operator's control traffic through a host HTTP proxy.
            trust_env=False,
        )
        # Event polling runs for up to 180 s per execution; a single stalled
        # poll should not kill the task.  Use a shorter connect timeout but
        # allow reads to block longer during Cortex tool calls (Qwen-VL +
        # vision-follower init).
        self._events_client = httpx.Client(
            base_url=self._edge_url,
            timeout=httpx.Timeout(connect=5.0, read=60.0, write=5.0, pool=5.0),
            headers={"Authorization": f"Bearer {self._token}"},
            trust_env=False,
        )
        self._lock = threading.Lock()
        self._active_command_id: str | None = None
        self._closed = False

    @staticmethod
    def _load_token(token_file: str | None, token: str | None) -> str:
        """Load a bearer token from an explicit value, file, or env.

        The tokens file maps token -> scope list. We pick a token that carries
        the ``task.submit`` scope so the operator can drive commands.
        """
        if token:
            return token
        if token_file and os.path.exists(token_file):
            with open(token_file, encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                for candidate, scopes in data.items():
                    if isinstance(scopes, list) and "task.submit" in scopes:
                        return str(candidate)
                return ""
            return str(data).strip()
        env_token = os.environ.get("UBROBOT_EDGE_TOKEN", "")
        if env_token:
            return env_token
        return ""

    @staticmethod
    def _new_nonce() -> str:
        return str(uuid4())

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def execute(
        self,
        task: str,
        *,
        on_feedback: Callable[[str], None] | None = None,
    ) -> str:
        """Submit ``task`` to Robot Edge and block until it reaches a terminal state.

        Feedback messages are forwarded to ``on_feedback`` (single string
        argument, matching the TaskBackend protocol). Returns the final message
        on success; raises RuntimeError on failure, cancellation, or disconnect.
        """
        if self._closed:
            raise RuntimeError("Robot Edge backend is closed")

        correlation_id = str(uuid4())
        command_id = self._submit_command(task, correlation_id)
        with self._lock:
            self._active_command_id = command_id
        try:
            return self._poll_events(command_id, on_feedback)
        finally:
            with self._lock:
                if self._active_command_id == command_id:
                    self._active_command_id = None

    def _submit_command(self, text: str, correlation_id: str) -> str:
        payload = {
            "text": text,
            "correlation_id": correlation_id,
            "operator_id": self._operator_id,
            "nonce": self._new_nonce(),
            "timestamp": self._now(),
        }
        try:
            response = self._client.post("/v1/commands", json=payload)
        except httpx.RequestError:
            raise RuntimeError(f"Could not connect to Robot Edge at {self._edge_url}")
        return self._parse_command_response(response)

    @staticmethod
    def _detail_from(response: httpx.Response) -> str:
        """Extract the server-provided detail string, sanitized.

        The Edge returns one detail per 409 reason: timestamp out of range,
        nonce already used, safety latched, or hardware authority disabled.
        Showing the real reason prevents misleading "replay or stale" claims.
        """
        try:
            data = response.json()
        except ValueError:
            return ""
        detail = data.get("detail") if isinstance(data, dict) else None
        if not isinstance(detail, str):
            return ""
        # The detail is generated server-side and never contains tokens;
        # bound its length anyway for log hygiene.
        return detail[:200]

    @staticmethod
    def _parse_command_response(response: httpx.Response) -> str:
        # Errors are sanitized: the bearer token never appears in messages.
        if response.status_code == 401:
            raise RuntimeError("Robot Edge authentication failed")
        if response.status_code == 403:
            raise RuntimeError("Robot Edge rejected the operator token scope")
        if response.status_code == 409:
            detail = RobotEdgeBackend._detail_from(response)
            if detail:
                raise RuntimeError(f"Robot Edge rejected command: {detail}")
            raise RuntimeError("Robot Edge rejected the request (replay or stale)")
        if response.status_code >= 400:
            raise RuntimeError(
                f"Robot Edge rejected command (HTTP {response.status_code})"
            )
        data = response.json()
        command_id = data.get("command_id")
        if not command_id:
            raise RuntimeError("Robot Edge accepted response missing command_id")
        return str(command_id)

    # Terminal/polling markers that should never be shown as the chat reply;
    # the real reply is the last substantive Cortex feedback (e.g.
    # "[No actions needed]. 你好！...").
    _NON_REPLY_MARKERS = (
        "Task complete!",
        "Post-execution:",
        "Plan aborted",
        "Command accepted",
        "Command cancelled",
        "Command failed",
    )

    @classmethod
    def _substantive_reply(cls, message: str) -> bool:
        return bool(message) and not any(
            marker in message for marker in cls._NON_REPLY_MARKERS
        )

    def _poll_events(
        self,
        command_id: str,
        on_feedback: Callable[[str], None] | None,
    ) -> str:
        last_sequence = 0
        # Cortex orchestration is slow: reasoning-model planning can take
        # 30-60 s, navigation timeout_sec is >= 60 s, and post-execution
        # async waits add more. Stay above the Edge's own 300 s result wait
        # so the Edge's terminal event always arrives first.
        deadline = time.monotonic() + 320.0
        last_substantive = ""
        _MAX_POLL_RETRIES = 3
        consecutive_errors = 0
        while time.monotonic() < deadline:
            if self._closed:
                raise RuntimeError("Robot Edge backend closed during execution")
            try:
                response = self._events_client.get(
                    "/v1/events", params={"after": last_sequence}
                )
                consecutive_errors = 0
            except httpx.RequestError:
                consecutive_errors += 1
                if consecutive_errors > _MAX_POLL_RETRIES:
                    raise RuntimeError("Lost connection to Robot Edge during execution")
                # Transient network glitch; back off briefly and retry.
                time.sleep(0.5 * consecutive_errors)
                continue
            if response.status_code == 401:
                raise RuntimeError("Robot Edge authentication failed during execution")
            if response.status_code >= 400:
                raise RuntimeError(
                    f"Robot Edge event stream error (HTTP {response.status_code})"
                )
            events = response.json().get("events", [])
            if not events:
                time.sleep(0.05)
                continue
            for event in events:
                sequence = int(event.get("sequence", last_sequence + 1))
                last_sequence = max(last_sequence, sequence)
                # The Edge stream is shared across commands (and safety
                # events). Replay from the cursor would otherwise let an
                # earlier command's terminal event end this execute().
                if event.get("command_id") != command_id:
                    continue
                state = event.get("state")
                message = str(event.get("message", "") or "")
                if message and on_feedback is not None:
                    on_feedback(message)
                if self._substantive_reply(message):
                    last_substantive = message
                if state == "succeeded":
                    return last_substantive or message or "Command completed"
                if state == "failed":
                    raise RuntimeError(message or "Command failed")
                if state == "cancelled":
                    raise RuntimeError(message or "Command cancelled")
        raise RuntimeError("Robot Edge command timed out")

    def get_robot_observation(self):
        """Fetch the latest camera frame from Robot Edge as a PIL image.

        Returns ``(navigation_image, None)`` — the console renders the
        robot's color camera in the "导航相机" tab; the manipulation/depth
        view has no dedicated source yet.
        """
        try:
            response = self._client.get("/v1/camera/frame")
        except httpx.RequestError:
            return None, None
        if response.status_code != 200:
            return None, None
        try:
            import io

            from PIL import Image as PILImage  # noqa: PLC0415

            image = PILImage.open(io.BytesIO(response.content))
            return image, None
        except Exception:
            return None, None

    def cancel_active(self) -> bool:
        """Cancel the active command. Returns True if the Edge acknowledged."""
        with self._lock:
            command_id = self._active_command_id
        if not command_id:
            return False
        payload = {
            "command_id": command_id,
            "correlation_id": str(uuid4()),
            "operator_id": self._operator_id,
            "nonce": self._new_nonce(),
            "timestamp": self._now(),
        }
        try:
            response = self._client.post(
                f"/v1/commands/{command_id}/cancel", json=payload
            )
        except httpx.RequestError:
            return False
        if response.status_code >= 400:
            return False
        with self._lock:
            if self._active_command_id == command_id:
                self._active_command_id = None
        return True

    def emergency_stop(self) -> bool:
        """Trigger the Edge safety stop (bypasses lease)."""
        payload = {
            "correlation_id": str(uuid4()),
            "operator_id": self._operator_id,
            "nonce": self._new_nonce(),
            "timestamp": self._now(),
        }
        try:
            response = self._client.post("/v1/safety/stop", json=payload)
        except httpx.RequestError:
            return False
        if response.status_code >= 400:
            return False
        with self._lock:
            self._active_command_id = None
        return True

    def close(self) -> None:
        """Close the HTTP clients. Idempotent and safe to call during execute."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._client.close()
        except Exception:
            logger.debug("robot edge client close failed", exc_info=True)
        try:
            self._events_client.close()
        except Exception:
            logger.debug("robot edge events client close failed", exc_info=True)
