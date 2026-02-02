#!/usr/bin/env python

import base64
import json
import logging
import time
from dataclasses import dataclass, field

import cv2

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient

class UnitreeGo2Robot:
    
    # unitree go2 dog
    def __init__(self):
        self.go2client = None
        ChannelFactoryInitialize(0, "eth0") # default net card
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
            self.go2client.SpeedLevel(-1) # slow 
            ret = self.go2client.Move(0.3,0,0)
            time.sleep(1)

            self.go2client.StopMove()
            return ret
