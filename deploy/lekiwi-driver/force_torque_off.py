#!/usr/bin/env python3
"""Verified torque-disable for the LeKiwi base motors.

Writes torque_enable=0 with per-motor acknowledge check and read-back,
and also verifies goal velocity is 0. Use ONLY with the driver container
paused (docker pause lekiwi-base-driver) so the bus is exclusive.
"""

import sys
import time

import serial

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
BAUD = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
MOTORS = {7: "left", 8: "back", 9: "right"}

INST_READ = 0x02
INST_WRITE = 0x03
TORQUE_ENABLE = 40
GOAL_VELOCITY = 46


def make_packet(motor_id, instruction, params):
    pkt = [0xFF, 0xFF, motor_id, len(params) + 2, instruction] + list(params)
    pkt.append(~sum(pkt[2:]) & 0xFF)
    return bytes(pkt)


def transact(ser, motor_id, instruction, params, expect_len):
    req = make_packet(motor_id, instruction, params)
    for _ in range(4):
        ser.reset_input_buffer()
        ser.write(req)
        time.sleep(0.003)
        header = ser.read(5)
        if len(header) < 5 or header[0] != 0xFF or header[1] != 0xFF or header[2] != motor_id:
            continue
        body_len = header[3] - 2
        if body_len != expect_len:
            continue
        rest = ser.read(body_len + 1)
        if len(rest) < body_len + 1:
            continue
        frame = header + rest
        if (~sum(frame[2:-1]) & 0xFF) != frame[-1]:
            continue
        return frame[5:5 + body_len], header[4]
    return None, None


def main():
    ser = serial.Serial(DEVICE, BAUD, timeout=0.1)
    ok = True
    for motor_id, name in sorted(MOTORS.items(), key=lambda kv: kv[1]):
        params, err = transact(ser, motor_id, INST_WRITE, [TORQUE_ENABLE, 0], 0)
        if params is None:
            print(f"motor {motor_id} ({name}): torque-off write NOT ACKNOWLEDGED")
            ok = False
            continue
        time.sleep(0.005)
        readback, err2 = transact(ser, motor_id, INST_READ, [TORQUE_ENABLE, 1], 1)
        goal, _ = transact(ser, motor_id, INST_READ, [GOAL_VELOCITY, 2], 2)
        goal_raw = (goal[0] | (goal[1] << 8)) if goal else None
        torque = readback[0] if readback else None
        status = "OK" if torque == 0 and goal_raw == 0 else "MISMATCH"
        if status != "OK":
            ok = False
        print(f"motor {motor_id} ({name}): torque_enable={torque} goal_raw={goal_raw} "
              f"err=0x{(err2 or 0):02x} -> {status}")
    ser.close()
    print("RESULT:", "all motors torque OFF, goal 0" if ok else "FAILED — inspect above")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
