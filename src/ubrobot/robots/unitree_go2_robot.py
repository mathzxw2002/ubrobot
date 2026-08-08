#!/usr/bin/env python
"""DEPRECATED direct Unitree Go2 SportClient movement (rollback/research only).

The production Go2 path drives the base via ``/cmd_vel`` through the Kompass
navigation chain (see docs/plans/2026-08-06-go2-piper-cortex-integration.md).
Direct SportClient motion is deprecated: importing this module does NOT
require unitree_sdk2py (imported lazily in ``__init__``) so workstations can
import the package without the SDK installed.
"""

import time
import warnings

warnings.warn(
    "ubrobot.robots.unitree_go2_robot.UnitreeGo2Robot is deprecated: "
    "Go2 motion must go through /cmd_vel (Kompass) instead of SportClient.",
    DeprecationWarning,
    stacklevel=2,
)


class UnitreeGo2Robot:
    # unitree go2 dog
    def __init__(self):
        # Lazy import: unitree_sdk2py is a robot-side dependency only.
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,  # noqa: PLC0415
        )
        from unitree_sdk2py.go2.sport.sport_client import SportClient  # noqa: PLC0415

        self.go2client = None
        ChannelFactoryInitialize(0, "eth0")  # default net card
        self.go2client = SportClient()
        self.go2client.SetTimeout(10.0)
        self.go2client.Init()
        # TODO set slow mode
        self.go2client.SpeedLevel(-1)

    def go2_robot_stop(self):
        if self.go2client is None:
            print("Go2 Sport Client NOT initialized!")
            return
        else:
            self.go2client.StopMove()

    def go2_robot_standup(self):
        if self.go2client is None:
            print("Go2 Sport Client NOT initialized!")
            return
        else:
            self.go2client.StandUp()

    def go2_robot_standdown(self):
        if self.go2client is None:
            print("Go2 Sport Client NOT initialized!")
            return
        else:
            self.go2client.StandDown()

    def go2_robot_move(self):
        if self.go2client is None:
            print("Go2 Sport Client NOT initialized!")
            return -1
        else:
            self.go2client.SpeedLevel(-1)  # slow
            ret = self.go2client.Move(0.3, 0, 0)
            time.sleep(1)

            self.go2client.StopMove()
            return ret
