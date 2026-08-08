"""Prometheus metrics for Robot Edge (P2, observability).

``prometheus_client`` is an optional runtime dependency: this module imports
it lazily so the fixture mode and workstation tests run without it. When the
client is available, ``metrics()`` returns the registry's render function;
otherwise ``/v1/metrics`` serves a 503 with an explanatory body.

Metrics exposed:

- ``ubrobot_edge_commands_total`` — submitted commands (by state)
- ``ubrobot_edge_lease_active`` — 1 while a lease is active
- ``ubrobot_edge_safety_latched`` — 1 while the safety latch is engaged
- ``ubrobot_edge_capability_available`` — per-capability availability gauge
- ``ubrobot_edge_estop_triggered`` — 1 while the physical E-stop latch is set
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("ubrobot.robot_edge.metrics")

_CLIENT = None
# (name, documentation) for the counters we lazily register.
_COMMANDS_TOTAL = ("ubrobot_edge_commands_total", "Commands submitted (by final state)")
_LEASE_ACTIVE = ("ubrobot_edge_lease_active", "1 while a navigation lease is active")
_SAFETY_LATCHED = ("ubrobot_edge_safety_latched", "1 while the safety latch is engaged")
_ESTOP_TRIGGERED = (
    "ubrobot_edge_estop_triggered",
    "1 while the physical E-stop latch is set",
)


class EdgeMetrics:
    """Registry wrapper that degrades gracefully when prometheus is missing."""

    def __init__(self) -> None:
        # Any: prometheus objects exist only when _enabled; guarded by checks.
        self._registry: Any = None
        self._commands: Any = None
        self._lease_active: Any = None
        self._safety_latched: Any = None
        self._estop_triggered: Any = None
        self._capability_gauge: Any = None
        self._enabled = False
        self._try_init()

    def _try_init(self) -> None:
        global _CLIENT
        try:
            import prometheus_client  # noqa: PLC0415 - optional dep

            _CLIENT = prometheus_client
            self._registry = prometheus_client.CollectorRegistry()
            name, doc = _COMMANDS_TOTAL
            self._commands = prometheus_client.Counter(
                name, doc, labelnames=("state",), registry=self._registry
            )
            name, doc = _LEASE_ACTIVE
            self._lease_active = prometheus_client.Gauge(
                name, doc, registry=self._registry
            )
            name, doc = _SAFETY_LATCHED
            self._safety_latched = prometheus_client.Gauge(
                name, doc, registry=self._registry
            )
            name, doc = _ESTOP_TRIGGERED
            self._estop_triggered = prometheus_client.Gauge(
                name, doc, registry=self._registry
            )
            self._capability_gauge = prometheus_client.Gauge(
                "ubrobot_edge_capability_available",
                "Capability availability (1 available, 0 otherwise)",
                labelnames=("capability",),
                registry=self._registry,
            )
            self._enabled = True
        except ModuleNotFoundError:
            logger.debug("prometheus_client not installed; /v1/metrics disabled")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def record_command(self, state: str) -> None:
        if self._enabled and self._commands is not None:
            self._commands.labels(state=state).inc()

    def set_lease_active(self, active: bool) -> None:
        if self._enabled and self._lease_active is not None:
            self._lease_active.set(1.0 if active else 0.0)

    def set_safety_latched(self, latched: bool) -> None:
        if self._enabled and self._safety_latched is not None:
            self._safety_latched.set(1.0 if latched else 0.0)

    def set_estop_triggered(self, triggered: bool) -> None:
        if self._enabled and self._estop_triggered is not None:
            self._estop_triggered.set(1.0 if triggered else 0.0)

    def set_capability(self, capability: str, available: bool) -> None:
        if self._enabled and self._capability_gauge is not None:
            self._capability_gauge.labels(capability=capability).set(
                1.0 if available else 0.0
            )

    def render(self) -> bytes:
        """Render the metrics payload; empty when prometheus is unavailable."""
        if not self._enabled or self._registry is None:
            return b""
        return _CLIENT.generate_latest(self._registry)  # type: ignore[union-attr]


# Module-level singleton for app wiring.
METRICS = EdgeMetrics()
