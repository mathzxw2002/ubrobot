# Robot Edge Hardware Inventory Report (M6, read-only)

- Date/time: 2026-08-02 (+08:00)
- Commit: `b65e5a2` (M5 complete, clean worktree)
- Machine role: robot-side host (Raspberry Pi, ssh alias `rasp_pi`)
- OS/architecture: Ubuntu 24.04.4 LTS, `aarch64`, Docker 29.5.1
- Execution mode: read-only inventory. No motion, no torque enable, no CAN
  activation, no `/cmd_vel`, no ROS goals, no SDK control session.
- Hardware authority: **false** (Robot Edge M5 fixture only; nothing started)
- Redaction: IPs, full serials, user names, and Wi-Fi credentials are
  omitted from this committed report per the M6 plan.

## Inventory (read-only)

| Item | Result |
|---|---|
| Host | Raspberry Pi, 4-core, 3.9 GiB RAM, rootfs 69% used / 36G free |
| Network | Wi-Fi (`wlan0`, LAN segment only); wired `eth0` DOWN |
| Docker containers | `lekiwi-base-driver` (Exited 0, 3 days), `emos` (Exited 137, 3 days), `emos-pre-fastdds-*` (Exited) |
| LeKiwi serial | `/dev/lekiwi-base -> ttyACM0` present; udev identity matches the committed `99-lekiwi-base.rules` (vendor/model/serial all match); group `dialout`, mode 0660 |
| RealSense | Intel RealSense D435i present on USB; **actual serial differs from the historical `config_piper.py` value** (historical value appears to belong to a different camera; actual serial recorded locally, not committed) |
| CAN (Piper) | **No CAN interfaces present**; `can-utils` installed on host but no `can0`; Piper is not connected to this host |
| GPIO | `gpiochip0..4` exported by the kernel; no E-stop/safety/watchdog systemd units or scripts found |
| udev | `99-lekiwi-base.rules` + `99-realsense-libusb.rules` installed |
| EMOS data dir | recipes `vision_depth_follower`, `vision_rgb_follower`, `my_first_recipe`; logs through 2026-07-30 |

## ROS Action comparison (deferred)

The planned Actions

- `/ubrobot/navigation/navigate_to_object`
- `/ubrobot/manipulation/grasp_object`

cannot be enumerated while both containers are stopped. This comparison is
part of M6 Task 10 (ROS-side read-only adapters) after the containers are
started with motion authority still disabled.

## Critical findings / stop conditions

1. **LeKiwi driver container is configured in torque-enabled state.**
   `docker inspect` shows `hardware_mode:=real`, `enable_real_hardware:=true`,
   and **`enable_motor_torque:=true`** with `/dev/lekiwi-base` mapped and
   `restart: no`. This is the plan's final hard-gate configuration. The
   container is stopped today, but **starting it would enable motor torque**.
   It must NOT be started until: a supervised lifted-wheel preflight plan is
   approved, the physical E-stop is verified, and the container is recreated
   in `mock` or torque-disabled `real` mode. This matches the plan stop
   condition "Piper torque is enabled unexpectedly" (here: LeKiwi).
2. **No physical E-stop evidence found** on the host (no unit, no GPIO
   binding, no watchdog service). ADR-0002 listed the E-stop choice as
   "to be confirmed". **M7 is blocked** until the owner confirms and wires a
   physical E-stop independent of the containers.
3. RealSense serial does not match the historical config; the committed
   `config_piper.py` value must be updated when the arm workstation is
   configured (separate from this host).
4. Piper is not connected to this host (no CAN); the chosen M6/M7 mobile-base
   profile remains **`lekiwi`**, consistent with the "exactly one base
   profile" constraint.

## Owner confirmation (2026-08-02, after initial inventory)

- LeKiwi real-hardware mode was previously exercised on the real base and
  confirmed normal by the owner; the container is intentionally left in its
  current (stopped, torque-enabled hard-gate) configuration. It remains
  **stopped** and must not be started without a supervised preflight.
- The LeKiwi base is wired and connected to the Raspberry Pi via the stable
  serial device.
- Piper is not connected to this host; mobile-base profile remains `lekiwi`.

## Evidence

- Raw output preserved in the terminal session; the committed report is
  sanitized. The preflight can be reproduced with
  `bash scripts/hardware/robot_edge_preflight.sh rasp_pi`.

## Acceptance

- [x] No motion, no torque enable, no authority granted.
- [x] All inventory steps were read-only.
- [x] Report sanitized (no IPs/serials/usernames/Wi-Fi credentials).
