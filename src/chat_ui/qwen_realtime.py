"""Qwen-Omni-Realtime WebSocket adapter for the neutral voice runtime."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
import json
import logging
import os
import queue
import threading
import time
from typing import Callable

try:
    from .voice_runtime import VoiceEvent, VoiceEventType
except ImportError:  # Direct-script compatibility.
    from voice_runtime import VoiceEvent, VoiceEventType


_SUBMIT_INTERACTION_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_interaction",
        "description": (
            "Submit every user request to the robot interaction runtime. "
            "This is the only available robot-facing operation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The complete user request as spoken.",
                }
            },
            "required": ["text"],
        },
    },
}

logger = logging.getLogger("ubrobot.voice.qwen")


@dataclass(frozen=True)
class QwenRealtimeConfig:
    api_key: str
    workspace_id: str
    model: str = "qwen3.5-omni-plus-realtime"
    voice: str = "Tina"
    region: str = "cn-beijing"
    connect_timeout_sec: float = 10.0
    session_timeout_sec: float = 1800.0
    proxy: str | bool | None = None

    def __post_init__(self):
        if self.connect_timeout_sec <= 0 or self.session_timeout_sec <= 0:
            raise ValueError("Qwen realtime timeouts must be positive")

    @classmethod
    def from_env(cls) -> "QwenRealtimeConfig":
        api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
        workspace_id = os.environ.get("DASHSCOPE_WORKSPACE_ID", "").strip()
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is required for Qwen realtime voice")
        if not workspace_id:
            raise RuntimeError(
                "DASHSCOPE_WORKSPACE_ID is required for Qwen realtime voice"
            )
        proxy_setting = os.environ.get(
            "UBROBOT_QWEN_REALTIME_PROXY", "direct"
        ).strip()
        if proxy_setting.lower() in {"", "direct", "none", "off"}:
            proxy: str | bool | None = None
        elif proxy_setting.lower() == "auto":
            proxy = True
        else:
            proxy = proxy_setting
        return cls(
            api_key=api_key,
            workspace_id=workspace_id,
            model=os.environ.get(
                "UBROBOT_QWEN_REALTIME_MODEL", "qwen3.5-omni-plus-realtime"
            ).strip(),
            voice=os.environ.get("UBROBOT_QWEN_REALTIME_VOICE", "Tina").strip(),
            region=os.environ.get(
                "UBROBOT_QWEN_REALTIME_REGION", "cn-beijing"
            ).strip(),
            session_timeout_sec=float(
                os.environ.get("UBROBOT_QWEN_REALTIME_SESSION_TIMEOUT_SEC", "1800")
            ),
            proxy=proxy,
        )

    @property
    def websocket_url(self) -> str:
        domains = {
            "cn-beijing": "cn-beijing.maas.aliyuncs.com",
            "ap-southeast-1": "ap-southeast-1.maas.aliyuncs.com",
        }
        try:
            domain = domains[self.region]
        except KeyError as exc:
            raise ValueError(
                "Qwen realtime region must be 'cn-beijing' or 'ap-southeast-1'"
            ) from exc
        return (
            f"wss://{self.workspace_id}.{domain}/api-ws/v1/realtime"
            f"?model={self.model}"
        )


class QwenOmniRealtimeProvider:
    """Raw WebSocket adapter with a single safe InteractionRuntime tool.

    Audio produced before the tool result is returned is deliberately dropped.
    This prevents a provider-side answer from bypassing Cortex.
    """

    name = "qwen-omni-realtime"

    def __init__(self, config: QwenRealtimeConfig):
        self.config = config
        self._sink: Callable[[VoiceEvent], None] | None = None
        self._outgoing: queue.Queue[dict | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._connected = threading.Event()
        self._startup_finished = threading.Event()
        self._stopped = threading.Event()
        self._startup_error: Exception | None = None
        self._lifecycle_lock = threading.RLock()
        self._awaiting_spoken_result = False
        self._tool_request_seen = False
        self._waiting_for_tool_phase_done = False
        self._active_request_id: str | None = None

    def start(self, event_sink: Callable[[VoiceEvent], None]) -> None:
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._sink = event_sink
            self._connected.clear()
            self._startup_finished.clear()
            self._stopped.clear()
            self._startup_error = None
            self._outgoing = queue.Queue()
            self._awaiting_spoken_result = False
            self._tool_request_seen = False
            self._waiting_for_tool_phase_done = False
            self._active_request_id = None
            self._thread = threading.Thread(
                target=self._thread_main,
                name="qwen-omni-realtime",
                daemon=True,
            )
            logger.info(
                "connecting model=%s region=%s proxy=%s",
                self.config.model,
                self.config.region,
                "auto" if self.config.proxy is True else (
                    "direct" if self.config.proxy is None else "explicit"
                ),
            )
            self._thread.start()

        deadline = time.monotonic() + self.config.connect_timeout_sec
        while time.monotonic() < deadline:
            if self._connected.wait(timeout=0.05):
                return
            if self._startup_finished.is_set():
                break
        error = self._startup_error
        self.stop()
        if error is not None:
            raise RuntimeError(f"Qwen realtime connection failed: {error}") from error
        raise TimeoutError("Qwen realtime WebSocket connection timed out")

    def stop(self) -> None:
        with self._lifecycle_lock:
            self._stopped.set()
            self._outgoing.put(None)
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        with self._lifecycle_lock:
            self._thread = None
            self._connected.clear()
        logger.info("session stopped")

    def push_audio(self, pcm: bytes) -> bool:
        if not self._connected.is_set() or self._stopped.is_set():
            return False
        self._send(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm).decode("ascii"),
            }
        )
        return True

    def complete_interaction(self, request_id: str, result: str) -> None:
        self._awaiting_spoken_result = True
        self._active_request_id = request_id
        self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": request_id,
                    "output": result,
                },
            }
        )
        self._send(
            {
                "type": "response.create",
                "response": {"modalities": ["text", "audio"]},
            }
        )

    def cancel_output(self) -> None:
        if self._connected.is_set() and not self._stopped.is_set():
            self._send({"type": "response.cancel"})

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run())
        except Exception as exc:
            self._startup_error = exc
            logger.exception("realtime session failed")
            self._emit(
                VoiceEvent(VoiceEventType.ERROR, error=f"Qwen realtime: {exc}")
            )
        finally:
            self._startup_finished.set()
            self._connected.clear()
            self._emit(VoiceEvent(VoiceEventType.DISCONNECTED))

    async def _run(self) -> None:
        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError(
                "websockets is required; install the ubrobot project dependencies"
            ) from exc

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "X-DashScope-OmniRealtime": "true",
        }
        async with websockets.connect(
            self.config.websocket_url,
            additional_headers=headers,
            proxy=self.config.proxy,
            open_timeout=self.config.connect_timeout_sec,
        ) as websocket:
            await websocket.send(json.dumps(self._session_update()))
            await asyncio.wait_for(
                self._run_connected(websocket),
                timeout=self.config.session_timeout_sec,
            )

    async def _run_connected(self, websocket) -> None:
        receiver = asyncio.create_task(self._receive_loop(websocket))
        sender = asyncio.create_task(self._send_loop(websocket))
        done, pending = await asyncio.wait(
            {receiver, sender}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        for task in done:
            task.result()

    async def _send_loop(self, websocket) -> None:
        while not self._stopped.is_set():
            message = await asyncio.to_thread(self._outgoing.get)
            if message is None:
                return
            await websocket.send(json.dumps(message, ensure_ascii=False))

    async def _receive_loop(self, websocket) -> None:
        async for raw in websocket:
            event = json.loads(raw)
            self._handle_server_event(event)

    def _handle_server_event(self, event: dict) -> None:
        event_type = event.get("type", "")
        if event_type == "session.updated":
            self._connected.set()
            self._startup_finished.set()
            logger.info("session ready")
            self._emit(VoiceEvent(VoiceEventType.CONNECTED))
        elif event_type == "conversation.item.input_audio_transcription.delta":
            text = event.get("text", "") + event.get("stash", "")
            logger.debug("partial transcript=%r", text)
            self._emit(
                VoiceEvent(
                    VoiceEventType.TRANSCRIPT_PARTIAL,
                    text=text,
                    request_id=event.get("item_id"),
                )
            )
        elif event_type == "conversation.item.input_audio_transcription.completed":
            logger.info("final transcript=%r", event.get("transcript", ""))
            self._emit(
                VoiceEvent(
                    VoiceEventType.TRANSCRIPT_FINAL,
                    text=event.get("transcript", ""),
                    request_id=event.get("item_id"),
                )
            )
        elif event_type == "input_audio_buffer.speech_started":
            self._emit(VoiceEvent(VoiceEventType.VAD_STARTED))
        elif event_type == "input_audio_buffer.speech_stopped":
            self._emit(VoiceEvent(VoiceEventType.VAD_STOPPED))
        elif event_type == "response.function_call_arguments.done":
            self._tool_request_seen = True
            self._waiting_for_tool_phase_done = True
            arguments = json.loads(event.get("arguments") or "{}")
            text = str(arguments.get("text") or "").strip()
            logger.info("interaction request call_id=%s text=%r", event.get("call_id"), text)
            self._emit(
                VoiceEvent(
                    VoiceEventType.INTERACTION_REQUEST,
                    text=text,
                    request_id=event.get("call_id"),
                )
            )
        elif event_type == "response.audio.delta":
            # Never play a provider response until Cortex has returned the tool
            # result for this turn.
            if self._awaiting_spoken_result:
                self._emit(
                    VoiceEvent(
                        VoiceEventType.AUDIO_CHUNK,
                        audio=base64.b64decode(event.get("delta", "")),
                    )
                )
        elif event_type == "response.done":
            if self._waiting_for_tool_phase_done:
                self._waiting_for_tool_phase_done = False
            elif self._awaiting_spoken_result:
                self._awaiting_spoken_result = False
                self._tool_request_seen = False
                self._emit(
                    VoiceEvent(
                        VoiceEventType.SPEECH_DONE,
                        request_id=self._active_request_id,
                    )
                )
                self._active_request_id = None
            elif not self._tool_request_seen:
                self.cancel_output()
                self._emit(
                    VoiceEvent(
                        VoiceEventType.ERROR,
                        error=(
                            "Qwen answered without submit_interaction; output was blocked"
                        ),
                    )
                )
        elif event_type == "error":
            error = event.get("error") or {}
            self._emit(
                VoiceEvent(
                    VoiceEventType.ERROR,
                    error=str(error.get("message") or error or "unknown provider error"),
                )
            )
        elif event_type == "connection.closed":
            self._emit(VoiceEvent(VoiceEventType.DISCONNECTED))
        elif event_type not in {
            "session.created",
            "response.created",
            "response.audio.done",
            "response.audio_transcript.delta",
            "response.audio_transcript.done",
            "input_audio_buffer.committed",
        }:
            logger.debug("provider event type=%s", event_type)

    def _session_update(self) -> dict:
        return {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "voice": self.config.voice,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "qwen3-asr-flash-realtime"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "silence_duration_ms": 800,
                },
                "instructions": (
                    "You are the speech interface for a robot. For every user "
                    "utterance, call submit_interaction exactly once with the "
                    "complete utterance. Never claim an action happened before "
                    "the tool result. Speak a concise summary of that result."
                ),
                "tools": [_SUBMIT_INTERACTION_TOOL],
            },
        }

    def _send(self, message: dict) -> None:
        self._outgoing.put(message)

    def _emit(self, event: VoiceEvent) -> None:
        if self._sink is not None:
            self._sink(event)
