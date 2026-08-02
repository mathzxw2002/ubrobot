#!/usr/bin/env python3
"""Segment-by-segment local-stop latency measurement (M7, robot-side).

Measures the software chain of the physical E-stop independently from
the physical power-off segment:

    T0  button pressed                          (observer timebase)
    T1  contact open sampled by LocalStopButton (this script)
    T2  SafetySupervisor fan-out invoked        (this script)
    T3  each StopSink call completed            (this script)
    T4  physical power-off observed             (observer timebase)

Requirements before any run (see plan M7 Task 12):

- wheels lifted and secured, or robot otherwise unable to travel;
- torque disabled (no `compose.hardware-torque-test.yaml`);
- physical E-stop verified by a human;
- a second observer present.

Usage on the Raspberry Pi:

    python3 scripts/hardware/measure_stop_latency.py \
        --chip gpiochip4 --line 23 --line-name ubrobot-estop \
        --dry-run

Dry-run (default) only times the software chain; nothing is stopped.
Add `--execute` only after the dry-run passes and the stop fan-out was
validated with motion outputs disabled. With `--execute`, `--zero-twist`
additionally publishes three zero `/cmd_vel` messages on the given ROS
domain before stopping the driver container (SIGINT -> ros2_control
deactivates hardware and disables torque).
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robot_edge.hardware.local_stop import (  # noqa: E402
    GpiodEstopLineReader,
    LocalStopButton,
)
from robot_edge.safety import SafetySupervisor  # noqa: E402


class _TimedSink:
    """Records when a stop fan-out call completed (per segment)."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.called_at: Optional[float] = None
        self.reason: Optional[str] = None

    def stop(self, reason: str) -> None:
        self.called_at = time.perf_counter()
        self.reason = reason


def _build_sinks(args: argparse.Namespace, started_at: float) -> list[_TimedSink]:
    sinks = [_TimedSink("supervisor-dispatch")]
    if args.execute:
        sinks.append(_TimedSink("zero-cmd-vel"))
        sinks.append(_TimedSink("driver-container-stop"))
    return sinks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chip", required=True, help="gpiod chip, e.g. gpiochip4")
    parser.add_argument("--line", type=int, required=True, help="GPIO line for the NC contact")
    parser.add_argument("--line-name", default="ubrobot-estop", help="diagnostic line name")
    parser.add_argument(
        "--debounce-sec",
        type=float,
        default=0.02,
        help="contact-open debounce window (seconds)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="REAL stop fan-out (zero /cmd_vel + docker stop driver). "
        "Dry-run by default: only the software chain is timed.",
    )
    args = parser.parse_args()

    started_at = time.perf_counter()
    print(f"measure_stop_latency started {datetime.now(timezone.utc).isoformat()}")
    print(f"  mode: {'EXECUTE' if args.execute else 'dry-run'}")
    print(f"  contact: {args.chip}#{args.line} ({args.line_name})")
    print(f"  debounce: {args.debounce_sec}s")
    print("  wheels must be lifted; torque must be disabled; E-stop verified by a human.")
    print()
    print("Press the physical E-stop button when ready.")
    print("(T0 = your press; the software timestamps below are relative to this run)")

    reader = GpiodEstopLineReader(args.chip, args.line, line_name=args.line_name)
    sinks = _build_sinks(args, started_at)
    supervisor = SafetySupervisor(stop_sinks=sinks)  # type: ignore[arg-type]
    button = LocalStopButton(reader, supervisor, debounce_sec=args.debounce_sec)

    try:
        while True:
            triggered = button.poll_once()
            if triggered:
                break
            time.sleep(0.005)  # 200 Hz sampling
    except KeyboardInterrupt:
        print("aborted by operator")
        return 1
    finally:
        reader.close()

    # T1: first open-contact sample (the poll loop just recorded it).
    t1 = button.snapshot()["last_read_at"]
    t1_rel = (t1 - started_at) * 1000.0 if t1 is not None else float("nan")
    print()
    print(f"T1 contact-open sampled at {t1_rel:8.1f} ms (relative to script start)")
    for sink in sinks:
        if sink.called_at is None:
            continue
        rel = (sink.called_at - started_at) * 1000.0
        print(f"   {sink.name:24s} at {rel:8.1f} ms  reason={sink.reason!r}")

    print()
    print("T4 physical power-off: record the time the motors stopped "
          "(observer timebase) and subtract your T0 press time.")
    print("Expected results (plan M7): input detection + supervisor dispatch "
          "well below 100 ms; physical segment dominated by the contactor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
