#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import json
import logging
import time
from dataclasses import dataclass, field

import cv2
import draccus
import zmq

from .config_piper import PiperConfig, PiperHostConfig
from .piper import Piper


@dataclass
class PiperServerConfig:
    """Configuration for the Piper host script."""

    robot: PiperConfig = field(default_factory=PiperConfig)
    host: PiperHostConfig = field(default_factory=PiperHostConfig)


class PiperHost:
    def __init__(self, config: PiperHostConfig):
        self.zmq_context = zmq.Context()
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_cmd_socket.bind(f"tcp://*:{config.port_zmq_cmd}")

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_observation_socket.bind(f"tcp://*:{config.port_zmq_observations}")

        self.connection_time_s = config.connection_time_s
        self.watchdog_timeout_ms = config.watchdog_timeout_ms
        self.max_loop_freq_hz = config.max_loop_freq_hz
        #
        self.robot = None

    def disconnect(self):
        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()

    def start_serving_teleoperation(self, cfg: PiperServerConfig):
        if self.robot is None:
            logging.info("Configuring Piper")
            self.robot = Piper(cfg.robot)
            logging.info("Connecting Piper")
            self.robot.connect()

        last_cmd_time = time.time()
        watchdog_active = False
        logging.info("Waiting for commands...")
        try:
            # Business logic
            start = time.perf_counter()
            duration = 0
            while duration < self.connection_time_s:
                loop_start_time = time.time()
                try:
                    msg = self.zmq_cmd_socket.recv_string(zmq.NOBLOCK)
                    data = dict(json.loads(msg))

                    print("-------------host,", data)
                    _action_sent = self.robot.send_action(data)
                    last_cmd_time = time.time()
                    watchdog_active = False
                except zmq.Again:
                    if not watchdog_active:
                        logging.warning("No command available")
                except Exception as e:
                    logging.error("Message fetching failed: %s", e)

                now = time.time()
                if (now - last_cmd_time > self.watchdog_timeout_ms / 1000) and not watchdog_active:
                    logging.warning(
                        f"Command not received for more than {self.watchdog_timeout_ms} milliseconds. Stopping the base."
                    )
                    watchdog_active = True
                    #TODO how to revise for piper
                    #robot.stop_base()

                last_observation = self.robot.get_observation()

                # Encode ndarrays to base64 strings
                for cam_key, _ in self.robot.cameras.items():
                    ret, buffer = cv2.imencode(
                        ".jpg", last_observation[cam_key], [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                    )
                    if ret:
                        last_observation[cam_key] = base64.b64encode(buffer).decode("utf-8")
                    else:
                        last_observation[cam_key] = ""

                    # if depth info exists
                    depth_key = f"{cam_key}_depth"
                    if last_observation[depth_key] is not None:
                        ret, buffer = cv2.imencode(
                            ".png", last_observation[depth_key]
                        )
                        if ret:
                            last_observation[depth_key] = base64.b64encode(buffer).decode("utf-8")
                        else:
                            last_observation[depth_key] = ""

                # Send the observation to the remote agent
                try:
                    self.zmq_observation_socket.send_string(json.dumps(last_observation), flags=zmq.NOBLOCK)
                except zmq.Again:
                    logging.info("Dropping observation, no client connected")

                # Ensure a short sleep to avoid overloading the CPU.
                elapsed = time.time() - loop_start_time

                time.sleep(max(1 / self.max_loop_freq_hz - elapsed, 0))
                duration = time.perf_counter() - start
            print("Cycle time reached.")

        except KeyboardInterrupt:
            print("Keyboard interrupt received. Exiting...")
        finally:
            print("Shutting down Piper Host.")
            self.robot.disconnect()
            self.disconnect()
        logging.info("Finished Piper cleanly")

    def get_robot_arm_observation_local(self):
        return self.robot.get_observation()

# comment this to avoid conflict with ubrobot robot arm serving
'''if __name__ == "__main__":
    logging.info("Starting HostAgent")
    cfg = PiperServerConfig()
    host = PiperHost(cfg.host)
    host.start_serving_teleoperation(cfg)'''
