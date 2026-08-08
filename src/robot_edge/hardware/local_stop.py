"""Physical local stop (E-stop auxiliary contact) binding (M7).

Binds a normally-closed (NC) emergency-stop contact to the existing
`SafetySupervisor` latch. Fail-closed by design:

- contact open (button pressed) -> stop
- contact wire broken / floating -> stop (NC to 3V3 with PULL_DOWN)
- reader raises -> treated as open -> stop

Workstation tests never import gpiod: they inject a fake
`EstopLineReader`. The gpiod-backed reader is constructed lazily (its
`gpiod` import happens in `__init__`), so importing this module is safe
anywhere and the hardware-SDK dependency stays robot-side only.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Callable, Optional

from robot_edge.safety import SafetySupervisor

__all__ = [
    "EstopLineReader",
    "GpiodEstopLineReader",
    "LocalStopButton",
    "EstopPoller",
]


class EstopLineReader(ABC):
    """Reads the E-stop auxiliary contact state.

    Implementations return `True` when the contact is closed (safe) and
    `False` when open (pressed / wiring fault). A reader must never
    raise on a fault; it returns `False` instead (fail-closed).
    """

    @abstractmethod
    def read(self) -> bool:
        """True = contact closed (safe); False = open (stop)."""

    @abstractmethod
    def describe(self) -> str:
        """Human-readable source description for events/logs."""


class GpiodEstopLineReader(EstopLineReader):
    """libgpiod-backed reader for a normally-closed auxiliary contact.

    Robot-side only: importing this class does not import gpiod; the
    import happens in `__init__` and fails naturally on workstations
    without libgpiod. Input uses internal PULL_DOWN so a broken wire
    reads low (= stop), matching the fail-closed requirement.
    """

    def __init__(
        self, chip: str, line: int, *, line_name: str = "ubrobot-estop"
    ) -> None:
        import gpiod  # deferred; robot-side dependency only

        self._chip = chip
        self._line = int(line)
        self._line_name = line_name
        self._request = gpiod.request_lines(
            chip,
            consumer="ubrobot-edge-estop",
            config={
                self._line: gpiod.LineSettings(
                    direction=gpiod.line.Direction.INPUT,
                    bias=gpiod.line.Bias.PULL_DOWN,
                )
            },
        )

    def read(self) -> bool:
        try:
            return bool(self._request.get_value(self._line))
        except Exception:
            # Any read fault fails closed: report "open contact".
            return False

    def describe(self) -> str:
        return f"gpiod:{self._chip}#{self._line}({self._line_name})"

    def close(self) -> None:
        self._request.release()


class LocalStopButton:
    """Debounced binding of a physical E-stop to SafetySupervisor.

    ``on_stop`` is the notification hook the app wires to the runtime so a
    physical stop cancels the active command and emits the critical event,
    not just the supervisor latch. When None (unit tests, standalone use),
    only ``supervisor.on_local_stop()`` is called.
    """

    def __init__(
        self,
        reader: EstopLineReader,
        supervisor: SafetySupervisor,
        *,
        debounce_sec: float = 0.02,
        clock: Optional[Callable[[], float]] = None,
        on_stop: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._reader = reader
        self._supervisor = supervisor
        self._debounce_sec = debounce_sec
        self._clock = clock or (lambda: datetime.now(timezone.utc).timestamp())
        self._on_stop = on_stop
        self._open_since: Optional[float] = None
        self._triggered = False
        self._last_read_at: Optional[float] = None
        self._last_read_ok: Optional[bool] = None
        self._read_count = 0

    @property
    def triggered(self) -> bool:
        return self._triggered

    def _fire(self, detail: str) -> None:
        if self._on_stop is not None:
            self._on_stop(detail)
        else:
            self._supervisor.on_local_stop(detail)

    def rearm(self) -> None:
        """Re-arm after an explicit authorized safety reset.

        Clears the triggered latch and the debounce window so the next poll
        re-samples the physical contact. If the contact is still open the
        next poll re-latches (fail-closed) instead of trusting the reset.
        """
        self._triggered = False
        self._open_since = None

    def poll_once(self) -> bool:
        """Sample the contact once; returns True when a stop is triggered.

        A brief glitch shorter than `debounce_sec` is ignored. Once the
        stop is triggered the supervisor latch owns the state; this
        method keeps returning True while the contact stays open so the
        caller can log the sustained condition.
        """
        now = self._clock()
        try:
            closed = self._reader.read()
        except Exception:
            closed = False  # fail-closed on any reader fault
        self._read_count += 1
        self._last_read_at = now
        self._last_read_ok = closed

        if self._triggered:
            return True
        if closed:
            self._open_since = None
            return False
        if self._open_since is None:
            self._open_since = now
            return False
        # Small epsilon absorbs float rounding (e.g. 1000.02 - 1000).
        if now - self._open_since >= self._debounce_sec - 1e-9:
            self._triggered = True
            self._fire(f"local stop: contact open ({self._reader.describe()})")
            return True
        return False

    def snapshot(self) -> dict:
        """Read-only diagnostic snapshot (no secrets, no file descriptors)."""
        return {
            "source": self._reader.describe(),
            "contact_closed": self._last_read_ok,
            "last_read_at": self._last_read_at,
            "read_count": self._read_count,
            "triggered": self._triggered,
        }


class EstopPoller:
    """Background poller driving a `LocalStopButton` (robot-side only)."""

    def __init__(self, button: LocalStopButton, *, interval_sec: float = 0.02) -> None:
        self._button = button
        self._interval_sec = interval_sec
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="ubrobot-estop-poller",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_sec):
            try:
                self._button.poll_once()
            except Exception:
                # The reader is fail-closed; a poller bug must not kill
                # the supervision thread, but keep trying.
                pass

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
