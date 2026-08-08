"""Robot Edge telemetry and capability streaming adapters.

These background clients pull serialized state from a remote Robot Edge and
publish it into the in-process TelemetryHub / CapabilityRegistry. They never
fabricate live data: if the Edge is unreachable, every channel is published as
``disconnected``. Hardware authority is granted only when the Edge reports
hardware mode AND the local config explicitly permits it.
"""

from __future__ import annotations

import threading
from typing import Any

import httpx

from ubrobot_contracts.capabilities import (
    CapabilityAvailability,
    CapabilityHealth,
    ExecutionMode,
)
from ubrobot_contracts.telemetry import TelemetryState

_CHANNELS = (
    "camera",
    "depth",
    "odometry",
    "joint_states",
    "navigation_lease",
    "capability_health",
)

# Map Edge capability string fields onto contract enums.
_AVAILABILITY = {member.value: member for member in CapabilityAvailability}
_HEALTH = {member.value: member for member in CapabilityHealth}
_EXECUTION_MODE = {member.value: member for member in ExecutionMode}


def _disconnected_value(channel: str) -> dict[str, Any]:
    """A channel value that the TelemetryHub reports as disconnected."""
    return {
        "channel": channel,
        "state": TelemetryState.DISCONNECTED.value,
        "available": False,
        "source": "robot-edge",
        "detail": "robot edge unreachable",
    }


class RobotEdgeTelemetryClient:
    """Background client that pulls telemetry snapshots from Robot Edge."""

    DEFAULT_POLL_INTERVAL = 1.0
    DEFAULT_BACKOFF_MAX = 5.0

    def __init__(
        self,
        edge_url: str,
        token: str,
        telemetry_hub: Any,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
    ):
        if not edge_url:
            raise ValueError("edge_url is required")
        if not token:
            raise RuntimeError(
                "Robot Edge telemetry token not configured. No default token is permitted."
            )
        self._edge_url = edge_url.rstrip("/")
        self._token = token
        self._telemetry_hub = telemetry_hub
        self._poll_interval = poll_interval
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._connected = False
        self._client = httpx.Client(
            base_url=self._edge_url,
            timeout=5.0,
            headers={"Authorization": f"Bearer {self._token}"},
            trust_env=False,
        )

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        try:
            self._client.close()
        except Exception:
            pass

    def close(self) -> None:
        self.stop()

    def mark_disconnected(self) -> None:
        """Publish a disconnected state for every edge-backed channel."""
        self._connected = False
        for channel in _CHANNELS:
            try:
                self._telemetry_hub.publish(channel, _disconnected_value(channel))
            except Exception:
                pass

    def poll_once(self) -> bool:
        """Pull one telemetry snapshot. Returns True on success.

        Public so tests can drive a single poll deterministically.
        """
        try:
            response = self._client.get("/v1/telemetry/snapshot")
        except httpx.RequestError:
            return False
        if response.status_code >= 400:
            return False
        self._process_telemetry(response.json())
        return True

    def _poll_loop(self) -> None:
        backoff = 0.1
        while not self._stop_event.is_set():
            ok = False
            try:
                ok = self.poll_once()
            except Exception:
                ok = False
            if ok:
                self._connected = True
                backoff = 0.1
            else:
                if self._connected:
                    self.mark_disconnected()
                backoff = min(backoff * 2, self.DEFAULT_BACKOFF_MAX)
            self._stop_event.wait(self._poll_interval if ok else backoff)

    def _process_telemetry(self, data: dict) -> None:
        channels = data.get("channels", {})
        for channel in _CHANNELS:
            snapshot = channels.get(channel)
            if not snapshot:
                # The Edge must report every channel explicitly; missing means
                # disconnected, never fabricated live data.
                self._telemetry_hub.publish(channel, _disconnected_value(channel))
                continue
            latest = snapshot.get("latest") or {}
            state = latest.get("state", TelemetryState.DISCONNECTED.value)
            value = latest.get("value")
            self._telemetry_hub.publish(
                channel,
                {
                    "channel": channel,
                    "state": state,
                    "available": state == TelemetryState.AVAILABLE.value,
                    "source": "robot-edge",
                    "value": value,
                },
            )


class RobotEdgeCapabilityClient:
    """Background client that pulls capability inventory from Robot Edge."""

    DEFAULT_POLL_INTERVAL = 5.0

    def __init__(
        self,
        edge_url: str,
        token: str,
        capability_registry: Any,
        local_hardware_permitted: bool = False,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
    ):
        if not edge_url:
            raise ValueError("edge_url is required")
        if not token:
            raise RuntimeError(
                "Robot Edge capability token not configured. No default token is permitted."
            )
        self._edge_url = edge_url.rstrip("/")
        self._token = token
        self._registry = capability_registry
        self._local_hardware_permitted = local_hardware_permitted
        self._poll_interval = poll_interval
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._client = httpx.Client(
            base_url=self._edge_url,
            timeout=5.0,
            headers={"Authorization": f"Bearer {self._token}"},
            trust_env=False,
        )

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        try:
            self._client.close()
        except Exception:
            pass

    def close(self) -> None:
        self.stop()

    def poll_once(self) -> bool:
        """Pull one capability snapshot. Returns True on success."""
        try:
            response = self._client.get("/v1/capabilities")
        except httpx.RequestError:
            return False
        if response.status_code >= 400:
            return False
        self._process_capabilities(response.json())
        return True

    def _poll_loop(self) -> None:
        backoff = 0.1
        while not self._stop_event.is_set():
            ok = False
            try:
                ok = self.poll_once()
            except Exception:
                ok = False
            if ok:
                backoff = 0.1
            else:
                backoff = min(backoff * 2, 5.0)
            self._stop_event.wait(self._poll_interval if ok else backoff)

    def _process_capabilities(self, data: dict) -> None:
        capabilities = data.get("capabilities", {})
        for name, cap_data in capabilities.items():
            if self._registry.get(name) is None:
                continue
            availability = _AVAILABILITY.get(
                cap_data.get("availability"), CapabilityAvailability.DISCONNECTED
            )
            health = _HEALTH.get(cap_data.get("health"), CapabilityHealth.UNKNOWN)
            execution_mode = _EXECUTION_MODE.get(cap_data.get("execution_mode"))
            edge_hardware_authority = bool(cap_data.get("hardware_authority", False))
            # Authority is granted only when the Edge reports hardware mode AND
            # local config explicitly permits it. Fixture/mock edges report
            # false, so authority stays false.
            actual_authority = (
                edge_hardware_authority
                and self._local_hardware_permitted
                and execution_mode == ExecutionMode.HARDWARE
            )
            self._registry.update(
                name,
                availability=availability,
                health=health,
                hardware_authority=actual_authority,
            )
