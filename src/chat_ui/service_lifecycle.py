"""Operator Console process diagnostics and bounded shutdown helpers."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import socket
from typing import Any


logger = logging.getLogger("ubrobot.operator_console.lifecycle")


@dataclass(frozen=True)
class PortInspection:
    host: str
    port: int
    available: bool
    pid: int | None = None
    process_name: str | None = None


class PortInUseError(RuntimeError):
    def __init__(self, inspection: PortInspection):
        owner = (
            f"PID {inspection.pid}"
            if inspection.pid is not None
            else "an unknown process"
        )
        if inspection.process_name:
            owner += f" ({inspection.process_name})"
        super().__init__(
            f"Cannot start Operator Console: {inspection.host}:{inspection.port} "
            f"is already listening in {owner}. Run "
            f"'scripts/operator_console.ps1 status -Port {inspection.port}' "
            "before starting another instance."
        )
        self.inspection = inspection


def inspect_port(host: str, port: int) -> PortInspection:
    """Return whether a TCP port can be bound and, when possible, its owner."""
    if not 0 < int(port) < 65536:
        raise ValueError("port must be between 1 and 65535")
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            probe.bind((host, int(port)))
        return PortInspection(host=host, port=int(port), available=True)
    except OSError:
        pid, process_name = _listener_owner(host, int(port))
        return PortInspection(
            host=host,
            port=int(port),
            available=False,
            pid=pid,
            process_name=process_name,
        )


def require_port_available(host: str, port: int) -> PortInspection:
    inspection = inspect_port(host, port)
    if not inspection.available:
        raise PortInUseError(inspection)
    return inspection


def shutdown_pipeline(pipeline: Any) -> None:
    """Best-effort, idempotent cleanup for app-owned runtime resources."""
    logger.info("operator runtime shutdown started")
    stop_event = getattr(pipeline, "stop", None)
    if stop_event is not None:
        stop_event.set()

    voice = getattr(pipeline, "voice_runtime", None)
    if voice is not None:
        try:
            voice.stop()
        except Exception:
            logger.exception("voice shutdown failed")

    task_runtime = getattr(pipeline, "task_runtime", None)
    if task_runtime is not None and task_runtime.active_task() is not None:
        try:
            task_runtime.cancel_active()
        except Exception:
            logger.exception("active task cancellation failed during shutdown")

    backend = getattr(pipeline, "backend", None)
    close = getattr(backend, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            logger.exception("backend close failed")
    elif backend is not None:
        cancel = getattr(backend, "cancel_active", None)
        if callable(cancel):
            try:
                cancel()
            except Exception:
                logger.exception("backend cancellation failed during shutdown")

    # Shut down Robot Edge telemetry and capability clients
    for name in ("edge_telemetry_client", "edge_capability_client"):
        client = getattr(pipeline, name, None)
        if client is not None:
            close_fn = getattr(client, "close", None) or getattr(client, "stop", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    logger.exception(f"{name} close failed")

    for name in ("tts_thread", "ffmpeg_thread"):
        worker = getattr(pipeline, name, None)
        if worker is not None and worker.is_alive():
            worker.join(timeout=2.0)
    logger.info("operator runtime shutdown complete")


def sanitized_capability_health(pipeline: Any) -> dict[str, Any]:
    snapshot = pipeline.telemetry_hub.snapshot().get("capability_health", {})
    value = snapshot.get("value", {})
    if not isinstance(value, dict):
        return {"status": "unknown"}
    try:
        from .adapters.telemetry import serialize_transport_value
    except ImportError:
        from adapters.telemetry import serialize_transport_value
    return serialize_transport_value(value)


def _listener_owner(host: str, port: int) -> tuple[int | None, str | None]:
    try:
        import psutil

        wildcard_hosts = {"0.0.0.0", "::"}
        for connection in psutil.net_connections(kind="tcp"):
            if connection.status != psutil.CONN_LISTEN or not connection.laddr:
                continue
            if connection.laddr.port != port:
                continue
            if connection.laddr.ip not in wildcard_hosts and host not in {
                connection.laddr.ip,
                "0.0.0.0",
                "::",
            }:
                continue
            pid = connection.pid
            if pid is None:
                return None, None
            try:
                return pid, psutil.Process(pid).name()
            except (psutil.Error, OSError):
                return pid, None
    except (ImportError, OSError):
        logger.debug("listener owner lookup unavailable", exc_info=True)
    return None, None
