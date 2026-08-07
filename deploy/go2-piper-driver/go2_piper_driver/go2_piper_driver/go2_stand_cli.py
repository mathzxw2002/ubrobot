"""Standalone Go2 posture CLI (runs in its own process, NO rclpy).

Executes StandUp / StandDown on the Go2 via the unitree_sdk2py SportClient.
This script must run as a SEPARATE process from any ROS 2 (rclpy) node:
initializing the unitree cyclonedds Python participant in a process that
already runs the RMW CycloneDDS participant segfaults (see
``go2_bridge_node.py`` docstring). No rclpy is imported here.

Exit codes: 0 = command accepted by the Go2; non-zero = failure.
"""

from __future__ import annotations

import argparse
import sys

try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.go2.sport.sport_client import SportClient
except Exception as exc:  # pragma: no cover - SDK absence
    print(f"unitree_sdk2py unavailable: {exc}", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stand", action="store_true", help="StandUp")
    group.add_argument("--sit", action="store_true", help="StandDown")
    parser.add_argument("--interface", default="eth0")
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    args = parser.parse_args()

    ChannelFactoryInitialize(0, args.interface)
    client = SportClient()
    client.SetTimeout(args.timeout_sec)
    client.Init()

    if args.stand:
        ret = client.StandUp()
    else:
        ret = client.StandDown()
    if ret != 0:
        print(f"sport command returned {ret}", file=sys.stderr)
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
