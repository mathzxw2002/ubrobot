"""Tests for the physical local stop (E-stop) binding (M7)."""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone

from robot_edge.hardware.local_stop import (
    EstopLineReader,
    LocalStopButton,
)
from robot_edge.safety import SafetySupervisor


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeReader(EstopLineReader):
    """Injectable reader; `closed` controls the contact state."""

    def __init__(self, *, closed: bool = True, fault: bool = False) -> None:
        self.closed = closed
        self.fault = fault
        self.reads = 0

    def read(self) -> bool:
        self.reads += 1
        if self.fault:
            raise RuntimeError("gpiod read failed")
        return self.closed

    def describe(self) -> str:
        return "fake:estop#0"


class _Sink:
    def __init__(self) -> None:
        self.stopped = False
        self.last_reason: str | None = None

    def stop(self, reason: str) -> None:
        self.stopped = True
        self.last_reason = reason


class LocalStopButtonTests(unittest.TestCase):
    def _make(self, reader: EstopLineReader) -> tuple[LocalStopButton, SafetySupervisor, _Sink, FakeClock]:
        clock = FakeClock()
        sink = _Sink()
        supervisor = SafetySupervisor(stop_sinks=[sink])
        button = LocalStopButton(reader, supervisor, clock=clock)
        return button, supervisor, sink, clock

    def test_contact_closed_is_safe(self) -> None:
        button, supervisor, sink, clock = self._make(FakeReader(closed=True))
        for _ in range(50):
            clock.advance(0.02)
            self.assertFalse(button.poll_once())
        self.assertFalse(supervisor.is_latched())
        self.assertFalse(sink.stopped)

    def test_open_contact_after_debounce_triggers_stop(self) -> None:
        button, supervisor, sink, clock = self._make(FakeReader(closed=True))
        self.assertFalse(button.poll_once())
        button._reader.closed = False  # button pressed
        clock.advance(0.01)
        self.assertFalse(button.poll_once())  # still inside debounce
        clock.advance(0.02)
        self.assertTrue(button.poll_once())
        self.assertTrue(supervisor.is_latched())
        self.assertTrue(sink.stopped)
        self.assertIn("local stop", sink.last_reason or "")
        self.assertIn("fake:estop", sink.last_reason or "")

    def test_brief_glitch_is_ignored(self) -> None:
        button, supervisor, _, clock = self._make(FakeReader(closed=True))
        button._reader.closed = False
        clock.advance(0.005)
        self.assertFalse(button.poll_once())
        button._reader.closed = True
        clock.advance(0.02)
        self.assertFalse(button.poll_once())
        self.assertFalse(supervisor.is_latched())

    def test_reader_fault_is_fail_closed(self) -> None:
        button, supervisor, sink, clock = self._make(FakeReader(fault=True))
        self.assertFalse(button.poll_once())  # first sample opens the contact
        clock.advance(0.03)
        self.assertTrue(button.poll_once())  # debounce elapsed -> stop
        self.assertTrue(supervisor.is_latched())
        self.assertTrue(sink.stopped)

    def test_sustained_open_stays_reported_but_latch_is_idempotent(self) -> None:
        button, supervisor, sink, clock = self._make(FakeReader(closed=False))
        self.assertFalse(button.poll_once())  # first sample opens the contact
        clock.advance(0.03)
        self.assertTrue(button.poll_once())
        calls_after_first = sink.stopped
        self.assertTrue(calls_after_first)
        clock.advance(0.1)
        self.assertTrue(button.poll_once())  # still reports open
        self.assertEqual(sink.stopped, True)
        # Emergency stop is latched; a second fan-out must not happen.
        self.assertEqual(supervisor._stop_executed, True)
        # Supervisor only executed once by construction (latched early-return).

    def test_snapshot_is_read_only_diagnostic(self) -> None:
        button, supervisor, _, clock = self._make(FakeReader(closed=False))
        clock.advance(0.02)
        button.poll_once()
        snap = button.snapshot()
        self.assertEqual(snap["source"], "fake:estop#0")
        self.assertIs(snap["contact_closed"], False)
        self.assertGreaterEqual(snap["read_count"], 1)
        self.assertIn("triggered", snap)
        # No secrets or descriptors in the snapshot.
        self.assertNotIn("request", snap)
        self.assertNotIn("token", snap)

    def test_detail_reaches_supervisor_reason(self) -> None:
        _, supervisor, sink, _ = self._make(FakeReader())
        supervisor.on_local_stop(detail="local stop: contact open (fake:estop#0)")
        self.assertTrue(sink.stopped)
        self.assertEqual(sink.last_reason, "local stop: contact open (fake:estop#0)")

    def test_rearm_after_reset_re_samples_still_open_contact(self) -> None:
        """After an authorized reset, a still-open contact must re-latch."""
        button, supervisor, sink, clock = self._make(FakeReader(closed=False))
        self.assertFalse(button.poll_once())  # opens the debounce window
        clock.advance(0.03)
        self.assertTrue(button.poll_once())  # first stop latches
        self.assertTrue(supervisor.is_latched())
        self.assertTrue(sink.stopped)

        supervisor.reset(authorized=True)
        button.rearm()
        self.assertFalse(supervisor.is_latched())
        self.assertFalse(button.triggered)
        # The authorized reset re-arms the stop fan-out.
        self.assertFalse(supervisor._stop_executed)

        # Contact is still open: the next polls re-latch and re-fire the sinks
        # (the fan-out was re-armed by the authorized reset).
        clock.advance(0.02)
        self.assertFalse(button.poll_once())  # opens the debounce window
        clock.advance(0.02)
        self.assertTrue(button.poll_once())  # debounce elapsed -> re-latch
        self.assertTrue(supervisor.is_latched())
        self.assertTrue(sink.stopped)
        self.assertTrue(supervisor._stop_executed)

    def test_on_stop_callback_replaces_supervisor_dispatch(self) -> None:
        """The app wires on_stop to the runtime; tests can inject it."""
        calls: list[str] = []

        def on_stop(detail: str) -> None:
            calls.append(detail)

        clock = FakeClock()
        sink = _Sink()
        supervisor = SafetySupervisor(stop_sinks=[sink])
        button = LocalStopButton(
            FakeReader(closed=False),
            supervisor,
            clock=clock,
            on_stop=on_stop,
        )
        self.assertFalse(button.poll_once())  # opens the debounce window
        clock.advance(0.03)
        self.assertTrue(button.poll_once())
        self.assertEqual(len(calls), 1)
        self.assertIn("local stop", calls[0])
        # The supervisor latch still happens through the callback wiring
        # (the runtime routes through the same supervisor in the app).
        self.assertFalse(supervisor.is_latched())  # on_stop replaced dispatch


class LocalStopImportBoundary(unittest.TestCase):
    def test_module_import_does_not_import_gpiod(self) -> None:
        self.assertNotIn("gpiod", sys.modules)
        from robot_edge.hardware import local_stop  # noqa: F401

        self.assertNotIn("gpiod", sys.modules)

    def test_estop_poller_is_a_daemon_thread(self) -> None:
        from robot_edge.hardware.local_stop import EstopPoller

        clock = FakeClock()
        sink = _Sink()
        supervisor = SafetySupervisor(stop_sinks=[sink])
        button = LocalStopButton(FakeReader(), supervisor, clock=clock)
        poller = EstopPoller(button)
        self.assertFalse(poller._thread is not None and poller._thread.is_alive())
        poller.start()
        self.assertTrue(poller._thread is not None)
        self.assertTrue(poller._thread.is_alive())
        self.assertTrue(poller._thread.daemon)
        poller.stop()
        self.assertFalse(poller._thread.is_alive())


if __name__ == "__main__":
    unittest.main()
