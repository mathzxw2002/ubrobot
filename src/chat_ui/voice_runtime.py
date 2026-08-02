"""Provider-neutral half-duplex voice session orchestration.

Cloud providers translate their native protocol into :class:`VoiceEvent` values.
The runtime deliberately depends on an interaction callback instead of Cortex or
robot capabilities, so a provider can never directly control hardware.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
import logging
import threading
from typing import Callable, Protocol
from uuid import uuid4


logger = logging.getLogger("ubrobot.voice.runtime")


class VoiceState(str, Enum):
    IDLE = "idle"
    CONNECTING = "connecting"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    EMERGENCY_STOPPED = "emergency_stopped"
    ERROR = "error"


class VoiceEventType(str, Enum):
    CONNECTED = "connected"
    TRANSCRIPT_PARTIAL = "transcript.partial"
    TRANSCRIPT_FINAL = "transcript.final"
    VAD_STARTED = "vad.started"
    VAD_STOPPED = "vad.stopped"
    INTERACTION_REQUEST = "interaction.request"
    AUDIO_CHUNK = "audio.chunk"
    SPEECH_DONE = "speech.done"
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass(frozen=True)
class VoiceEvent:
    event_type: VoiceEventType
    text: str = ""
    audio: bytes = b""
    request_id: str | None = None
    error: str | None = None


class VoiceProvider(Protocol):
    name: str

    def start(self, event_sink: Callable[[VoiceEvent], None]) -> None: ...

    def stop(self) -> None: ...

    def push_audio(self, pcm: bytes) -> bool: ...

    def complete_interaction(self, request_id: str, result: str) -> None: ...

    def cancel_output(self) -> None: ...


@dataclass(frozen=True)
class VoiceSnapshot:
    state: VoiceState
    provider: str
    session_active: bool
    session_id: str | None
    transcript_partial: str
    transcript_final: str
    last_reply: str
    last_error: str | None
    vad_active: bool
    microphone_level: float
    playback_pending: bool
    updated_at: datetime

    def to_dict(self):
        value = asdict(self)
        value["state"] = self.state.value
        value["updated_at"] = self.updated_at.isoformat()
        return value


class VoiceSessionManager:
    """Run one half-duplex voice session above InteractionRuntime.

    Normal audio is accepted only while LISTENING. Emergency stop is not a
    voice-provider event: it has a separate local/UI path and is accepted in
    every state.
    """

    def __init__(
        self,
        provider: VoiceProvider,
        *,
        interaction_handler: Callable[[str], str],
        contextual_interaction_handler: Callable[[str, str], str] | None = None,
        emergency_stop_handler: Callable[[str], bool],
        audio_sink: Callable[[bytes], None] | None = None,
        control_sink: Callable[[str], None] | None = None,
        event_publisher: Callable[..., object] | None = None,
    ):
        self._provider = provider
        self._interaction_handler = interaction_handler
        self._contextual_interaction_handler = contextual_interaction_handler
        self._emergency_stop_handler = emergency_stop_handler
        self._audio_sink = audio_sink
        self._control_sink = control_sink
        self._event_publisher = event_publisher
        self._lock = threading.RLock()
        self._state = VoiceState.IDLE
        self._active = False
        self._partial = ""
        self._final = ""
        self._reply = ""
        self._error: str | None = None
        self._updated_at = datetime.now(timezone.utc)
        self._interaction_worker: threading.Thread | None = None
        self._session_id: str | None = None
        self._generation = 0
        self._vad_active = False
        self._microphone_level = 0.0
        self._playback_pending = False
        self._provider_speech_done = False
        self._active_trace_id: str | None = None

    def start(self) -> VoiceSnapshot:
        with self._lock:
            if self._active:
                return self.snapshot()
            self._generation += 1
            generation = self._generation
            self._session_id = uuid4().hex
            self._active = True
            self._partial = ""
            self._final = ""
            self._reply = ""
            self._error = None
            self._vad_active = False
            self._microphone_level = 0.0
            self._playback_pending = False
            self._provider_speech_done = False
            self._active_trace_id = None
            self._set_state(VoiceState.CONNECTING)
            logger.info("state=%s provider=%s", self._state.value, self._provider.name)
        try:
            self._provider.start(
                lambda event: self.handle_provider_event(
                    event,
                    generation=generation,
                )
            )
        except Exception as exc:
            with self._lock:
                self._active = False
                self._error = str(exc)
                self._set_state(VoiceState.ERROR)
            raise
        return self.snapshot()

    def stop(self) -> VoiceSnapshot:
        with self._lock:
            self._generation += 1
        self._provider.cancel_output()
        self._provider.stop()
        with self._lock:
            self._active = False
            self._vad_active = False
            self._microphone_level = 0.0
            self._playback_pending = False
            self._provider_speech_done = False
            self._active_trace_id = None
            self._set_state(VoiceState.IDLE)
            logger.info("state=%s provider=%s", self._state.value, self._provider.name)
        return self.snapshot()

    def push_audio(self, pcm: bytes) -> bool:
        if not pcm:
            return False
        with self._lock:
            if not self._active or self._state != VoiceState.LISTENING:
                return False
        return bool(self._provider.push_audio(pcm))

    def emergency_stop(self, source: str = "voice-keyword") -> bool:
        # Cancel speech first so the stop acknowledgement cannot be hidden by
        # buffered audio. Robot cancellation then uses the independent safety
        # path supplied by TaskRuntime/its backend.
        self._provider.cancel_output()
        acknowledged = bool(self._emergency_stop_handler(source))
        with self._lock:
            self._generation += 1
            self._active = False
            self._set_state(VoiceState.EMERGENCY_STOPPED)
            logger.warning("state=%s source=%s", self._state.value, source)
        return acknowledged

    def set_audio_sink(self, audio_sink: Callable[[bytes], None] | None) -> None:
        with self._lock:
            self._audio_sink = audio_sink

    def set_control_sink(self, control_sink: Callable[[str], None] | None) -> None:
        with self._lock:
            self._control_sink = control_sink

    def update_microphone_level(self, level: float) -> None:
        normalized = max(0.0, min(1.0, float(level)))
        with self._lock:
            if not self._active:
                return
            self._microphone_level = normalized
            self._touch()
            self._publish(
                "voice.microphone_level",
                payload={"level": normalized},
            )

    def playback_finished(self) -> bool:
        with self._lock:
            was_pending = self._playback_pending
            self._playback_pending = False
            self._publish("voice.playback.done", payload={})
            if (
                self._active
                and self._provider_speech_done
                and self._state == VoiceState.SPEAKING
            ):
                self._provider_speech_done = False
                self._active_trace_id = None
                self._set_state(VoiceState.LISTENING)
            return was_pending

    def handle_provider_event(
        self,
        event: VoiceEvent,
        *,
        generation: int | None = None,
    ) -> None:
        with self._lock:
            if generation is not None and generation != self._generation:
                logger.debug("ignored stale voice event generation=%s", generation)
                return
        if event.event_type == VoiceEventType.CONNECTED:
            with self._lock:
                if self._active:
                    self._set_state(VoiceState.LISTENING)
                    logger.info("state=%s", self._state.value)
            return
        if event.event_type in {VoiceEventType.VAD_STARTED, VoiceEventType.VAD_STOPPED}:
            with self._lock:
                if self._active:
                    self._vad_active = event.event_type == VoiceEventType.VAD_STARTED
                    self._touch()
                    self._publish(
                        "voice.vad",
                        correlation_id=self._active_trace_id or event.request_id,
                        payload={"active": self._vad_active},
                    )
            return
        if event.event_type == VoiceEventType.TRANSCRIPT_PARTIAL:
            with self._lock:
                if self._state == VoiceState.LISTENING:
                    self._partial = event.text
                    self._touch()
                    self._publish(
                        "voice.transcript.partial",
                        correlation_id=event.request_id,
                        payload={"text": event.text},
                    )
                    logger.debug("partial transcript=%r", event.text)
            return
        if event.event_type == VoiceEventType.TRANSCRIPT_FINAL:
            with self._lock:
                if self._state == VoiceState.LISTENING:
                    self._final = event.text
                    self._partial = ""
                    self._touch()
                    self._publish(
                        "voice.transcript.final",
                        correlation_id=event.request_id,
                        payload={"text": event.text},
                    )
                    logger.info("final transcript=%r", event.text)
            return
        if event.event_type == VoiceEventType.INTERACTION_REQUEST:
            self._begin_interaction_request(event)
            return
        if event.event_type == VoiceEventType.AUDIO_CHUNK:
            with self._lock:
                allowed = self._active and self._state == VoiceState.SPEAKING
                sink = self._audio_sink
                if allowed and sink is not None and event.audio:
                    self._playback_pending = True
                    self._touch()
            if allowed and sink is not None and event.audio:
                sink(event.audio)
            return
        if event.event_type == VoiceEventType.SPEECH_DONE:
            control_sink = None
            with self._lock:
                if self._active and self._state == VoiceState.SPEAKING:
                    self._provider_speech_done = True
                    self._publish(
                        "voice.provider_speech_done",
                        correlation_id=self._active_trace_id or event.request_id,
                        payload={"playback_pending": self._playback_pending},
                    )
                    if not self._playback_pending:
                        self._provider_speech_done = False
                        self._active_trace_id = None
                        self._set_state(VoiceState.LISTENING)
                        logger.info("state=%s", self._state.value)
                    control_sink = self._control_sink
            if control_sink is not None:
                control_sink("provider.speech_done")
            return
        if event.event_type == VoiceEventType.DISCONNECTED:
            control_sink = None
            with self._lock:
                self._active = False
                if self._state != VoiceState.ERROR:
                    self._set_state(VoiceState.IDLE)
                logger.info("provider disconnected state=%s", self._state.value)
                control_sink = self._control_sink
            if control_sink is not None:
                control_sink("provider.disconnected")
            return
        if event.event_type == VoiceEventType.ERROR:
            control_sink = None
            with self._lock:
                self._error = event.error or event.text or "voice provider error"
                self._active = False
                self._set_state(VoiceState.ERROR)
                logger.error("state=error error=%s", self._error)
                control_sink = self._control_sink
            if control_sink is not None:
                control_sink("provider.error")

    def snapshot(self) -> VoiceSnapshot:
        with self._lock:
            return VoiceSnapshot(
                state=self._state,
                provider=self._provider.name,
                session_active=self._active,
                session_id=self._session_id,
                transcript_partial=self._partial,
                transcript_final=self._final,
                last_reply=self._reply,
                last_error=self._error,
                vad_active=self._vad_active,
                microphone_level=self._microphone_level,
                playback_pending=self._playback_pending,
                updated_at=self._updated_at,
            )

    def _begin_interaction_request(self, event: VoiceEvent) -> None:
        request_id = event.request_id
        text = event.text.strip()
        with self._lock:
            if not self._active or self._state != VoiceState.LISTENING:
                return
            if not request_id or not text:
                self._error = "provider emitted an invalid interaction request"
                self._set_state(VoiceState.ERROR)
                return
            self._final = text
            self._partial = ""
            trace_id = f"{self._session_id}:{request_id}"
            self._active_trace_id = trace_id
            self._publish(
                "voice.interaction.request",
                correlation_id=trace_id,
                payload={"text": text},
            )
            self._set_state(VoiceState.THINKING)
            logger.info("state=%s request_id=%s", self._state.value, request_id)
        worker = threading.Thread(
            target=self._execute_interaction_request,
            args=(request_id, text, trace_id),
            name="voice-interaction",
            daemon=True,
        )
        with self._lock:
            self._interaction_worker = worker
        worker.start()

    def _execute_interaction_request(
        self,
        request_id: str,
        text: str,
        trace_id: str,
    ) -> None:
        try:
            if self._contextual_interaction_handler is not None:
                reply = self._contextual_interaction_handler(text, trace_id)
            else:
                reply = self._interaction_handler(text)
            with self._lock:
                if not self._active:
                    return
                self._reply = reply
                self._set_state(VoiceState.SPEAKING)
                logger.info("state=%s request_id=%s", self._state.value, request_id)
            self._provider.complete_interaction(request_id, reply)
        except Exception as exc:
            with self._lock:
                self._error = str(exc)
                self._set_state(VoiceState.ERROR)
            self._provider.cancel_output()

    def _set_state(self, state: VoiceState) -> None:
        self._state = state
        self._touch()
        self._publish("voice.state", payload={"state": state.value})

    def _touch(self) -> None:
        self._updated_at = datetime.now(timezone.utc)

    def _publish(
        self,
        kind: str,
        *,
        payload: dict,
        correlation_id: str | None = None,
    ) -> None:
        if self._event_publisher is None:
            return
        try:
            self._event_publisher(
                kind=kind,
                source="voice_runtime",
                correlation_id=correlation_id or self._session_id,
                payload={
                    **payload,
                    "session_id": self._session_id,
                    "provider": self._provider.name,
                },
            )
        except Exception:
            logger.exception("voice event publication failed kind=%s", kind)


class MockVoiceProvider:
    """Deterministic provider used by hardware/cloud-free tests."""

    name = "mock"

    def __init__(self):
        self.event_sink: Callable[[VoiceEvent], None] | None = None
        self.audio_inputs: list[bytes] = []
        self.completed: list[tuple[str, str]] = []
        self.cancel_count = 0

    def start(self, event_sink: Callable[[VoiceEvent], None]) -> None:
        self.event_sink = event_sink
        event_sink(VoiceEvent(VoiceEventType.CONNECTED))

    def stop(self) -> None:
        if self.event_sink is not None:
            self.event_sink(VoiceEvent(VoiceEventType.DISCONNECTED))

    def push_audio(self, pcm: bytes) -> bool:
        self.audio_inputs.append(pcm)
        return True

    def complete_interaction(self, request_id: str, result: str) -> None:
        self.completed.append((request_id, result))

    def cancel_output(self) -> None:
        self.cancel_count += 1

    def emit(self, event: VoiceEvent) -> None:
        if self.event_sink is None:
            raise RuntimeError("mock voice provider is not started")
        self.event_sink(event)


class DisabledVoiceProvider:
    name = "disabled"

    def start(self, event_sink: Callable[[VoiceEvent], None]) -> None:
        raise RuntimeError(
            "voice is disabled; set UBROBOT_VOICE_PROVIDER=qwen or mock"
        )

    def stop(self) -> None:
        return None

    def push_audio(self, pcm: bytes) -> bool:
        return False

    def complete_interaction(self, request_id: str, result: str) -> None:
        raise RuntimeError("voice is disabled")

    def cancel_output(self) -> None:
        return None
