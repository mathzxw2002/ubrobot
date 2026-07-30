#!/usr/bin/env python3
"""Bisect which activation write flips STS3215 torque_enable to 1.

Reproduces the exact FeetechBus::configure_velocity_mode() +
stop_and_disable() + preflight write() loop sequence one step at a time,
reading back torque_enable (addr 40) after every step.

Run ONLY with the driver container paused (exclusive bus).
Safe: goal velocity is always 0; worst case wheels lock in place.
"""

import sys
import time

import serial

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
BAUD = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
MOTOR_IDS = [8, 9, 7]  # back, right, left (C++ MotorIds order)

INST_READ = 0x02
INST_WRITE = 0x03
INST_SYNC_READ = 0x82
INST_SYNC_WRITE = 0x83
BROADCAST = 0xFE

TORQUE_ENABLE = 40
GOAL_VELOCITY = 46
LOCK = 55
RETURN_DELAY = 7
MAX_ACCEL = 85
ACCEL = 41
OP_MODE = 33


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
        return frame[5:5 + body_len]
    return None


def read_torque(ser):
    states = {}
    for mid in MOTOR_IDS:
        r = transact(ser, mid, INST_READ, [TORQUE_ENABLE, 1], 1)
        states[mid] = r[0] if r else "?"
    return states


def write_reg(ser, mid, addr, data):
    return transact(ser, mid, INST_WRITE, [addr] + list(data), 0) is not None


def sync_write_goal(ser, values):
    # C++ FeetechBus::write_velocities: sync write addr 46 len 2 per motor
    params = [GOAL_VELOCITY, 2]
    for mid, raw in zip(MOTOR_IDS, values):
        params += [mid, raw & 0xFF, (raw >> 8) & 0xFF]
    ser.reset_input_buffer()
    ser.write(make_packet(BROADCAST, INST_SYNC_WRITE, params))
    time.sleep(0.005)


def sync_read_velocities(ser):
    params = [58, 2] + list(MOTOR_IDS)
    ser.reset_input_buffer()
    ser.write(make_packet(BROADCAST, INST_SYNC_READ, params))
    time.sleep(0.005)
    for _ in MOTOR_IDS:
        ser.read(8)


def checkpoint(ser, label):
    time.sleep(0.02)
    states = read_torque(ser)
    print(f"{label:55s} torque={states}")
    return states


def main():
    ser = serial.Serial(DEVICE, BAUD, timeout=0.1)
    print("step 0: force torque off + goal 0 on all motors")
    for mid in MOTOR_IDS:
        write_reg(ser, mid, TORQUE_ENABLE, [0])
        write_reg(ser, mid, GOAL_VELOCITY, [0, 0])
    base = checkpoint(ser, "baseline after force-off")
    if any(v != 0 for v in base.values()):
        print("torque not off at baseline, aborting bisect")
        sys.exit(2)

    print("\n--- configure_velocity_mode() writes, per motor ---")
    for mid in MOTOR_IDS:
        write_reg(ser, mid, TORQUE_ENABLE, [0])
        checkpoint(ser, f"motor {mid}: write torque=0")
        write_reg(ser, mid, LOCK, [0])
        checkpoint(ser, f"motor {mid}: write lock=0 (EEPROM unlock)")
        write_reg(ser, mid, RETURN_DELAY, [0])
        checkpoint(ser, f"motor {mid}: write return_delay=0")
        write_reg(ser, mid, MAX_ACCEL, [254])
        checkpoint(ser, f"motor {mid}: write max_accel=254")
        write_reg(ser, mid, ACCEL, [254])
        checkpoint(ser, f"motor {mid}: write accel=254")
        write_reg(ser, mid, OP_MODE, [1])
        checkpoint(ser, f"motor {mid}: write op_mode=1 (velocity)")

    print("\n--- zero_goal_registers_verified(): individual goal=0 writes ---")
    for mid in MOTOR_IDS:
        write_reg(ser, mid, GOAL_VELOCITY, [0, 0])
        checkpoint(ser, f"motor {mid}: individual write goal=0")

    print("\n--- write_velocities({0,0,0}): sync write goal=0 ---")
    sync_write_goal(ser, [0, 0, 0])
    checkpoint(ser, "sync write goal=0 (all motors)")

    print("\n--- stop_and_disable() torque=0 + lock=0 ---")
    for mid in MOTOR_IDS:
        write_reg(ser, mid, TORQUE_ENABLE, [0])
        write_reg(ser, mid, LOCK, [0])
    checkpoint(ser, "after explicit torque=0 + lock=0")

    print("\n--- preflight write() loop: repeated sync goal=0 ---")
    for i in range(3):
        sync_write_goal(ser, [0, 0, 0])
        checkpoint(ser, f"sync goal=0 cycle {i + 1}")

    print("\n--- sync read velocities ---")
    sync_read_velocities(ser)
    checkpoint(ser, "after sync read")
    ser.close()


if __name__ == "__main__":
    main()
