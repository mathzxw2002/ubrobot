#!/usr/bin/env python3
"""Read-only Feetech STS3215 register dump for the LeKiwi base motors.

Safety: this script performs READ instructions only. It never writes
registers and never enables torque. Run before any real-mode session to
satisfy the 2026-07-29 open item (goal/present/error register dump).

Usage: python3 dump_motor_registers.py [device] [baud]
"""

import sys
import time

import serial

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
BAUD = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000

MOTORS = {7: "left", 8: "back", 9: "right"}

INST_READ = 0x02

# (name, address, length, decode)  decode: 'u16', 's15' (sign-magnitude), 'u8'
REGISTERS = [
    ("model_number", 3, 2, "u16"),
    ("operating_mode", 33, 1, "u8"),
    ("torque_enable", 40, 1, "u8"),
    ("goal_velocity", 46, 2, "s15"),
    ("eeprom_lock", 55, 1, "u8"),
    ("present_position", 56, 2, "u16"),
    ("present_velocity", 58, 2, "s15"),
    ("present_load", 60, 2, "s15"),
    ("present_voltage", 62, 1, "u8"),
    ("present_temperature", 63, 1, "u8"),
    ("max_acceleration", 85, 1, "u8"),
]

STATUS_ERROR_BITS = {
    0: "voltage", 1: "sensor", 2: "temperature",
    3: "current", 4: "angle", 5: "overload",
}


def make_packet(motor_id, instruction, params):
    pkt = [0xFF, 0xFF, motor_id, len(params) + 2, instruction] + list(params)
    pkt.append(~sum(pkt[2:]) & 0xFF)
    return bytes(pkt)


def read_register(ser, motor_id, address, length, retries=3):
    """Return (params bytes, error byte) or (None, None) on failure."""
    req = make_packet(motor_id, INST_READ, [address, length])
    for _ in range(retries):
        ser.reset_input_buffer()
        ser.write(req)
        time.sleep(0.002)
        header = ser.read(5)
        if len(header) < 5 or header[0] != 0xFF or header[1] != 0xFF:
            continue
        body_len = header[3] - 2  # params length
        if body_len < 0 or header[2] != motor_id:
            continue
        rest = ser.read(body_len + 1)
        if len(rest) < body_len + 1:
            continue
        frame = header + rest
        if (~sum(frame[2:-1]) & 0xFF) != frame[-1]:
            continue
        return frame[5:5 + body_len], header[4]
    return None, None


def decode(params, kind):
    raw = params[0] | (params[1] << 8) if len(params) == 2 else params[0]
    if kind == "s15":
        mag = raw & 0x7FFF
        return -mag if raw & 0x8000 else mag
    return raw


def main():
    ser = serial.Serial(DEVICE, BAUD, timeout=0.1)
    print(f"device={DEVICE} baud={BAUD}  (READ-ONLY dump, no writes)")
    print()
    any_response = False
    for motor_id, name in sorted(MOTORS.items(), key=lambda kv: kv[1]):
        params, err = read_register(ser, motor_id, 3, 2)
        if params is None:
            print(f"motor {motor_id} ({name}): NO RESPONSE (unpowered or absent)")
            continue
        any_response = True
        model = decode(params, "u16")
        model_note = "OK" if model == 777 else f"UNEXPECTED (want 777)"
        print(f"motor {motor_id} ({name}): model={model} {model_note}")
        for reg_name, addr, length, kind in REGISTERS[1:]:
            params, err = read_register(ser, motor_id, addr, length)
            if params is None:
                print(f"  {reg_name:20s}: READ FAILED")
                continue
            value = decode(params, kind)
            unit = ""
            if reg_name == "present_voltage":
                unit = f"  ({value / 10.0:.1f} V)"
            elif reg_name == "goal_velocity" or reg_name == "present_velocity":
                unit = f"  ({value * 2 * 3.141592653589793 / 4096.0:+.4f} rad/s)"
            flags = ""
            if err:
                bits = [b for b, n in STATUS_ERROR_BITS.items() if err & (1 << b)]
                flags = f"  ERROR=0x{err:02x} [{','.join(bits)}]"
            print(f"  {reg_name:20s}: {value}{unit}{flags}")
        print()
    ser.close()
    if not any_response:
        print("RESULT: no motor responded — check motor power on the bus.")
        sys.exit(2)
    print("RESULT: dump complete.")


if __name__ == "__main__":
    main()
