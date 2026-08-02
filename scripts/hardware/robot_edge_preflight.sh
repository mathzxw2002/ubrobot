#!/usr/bin/env bash
# Robot Edge read-only preflight inventory.
#
# M6 Task 9: capture robot-side inventory WITHOUT motion authority.
# This script ONLY reads state: it never activates CAN, never enables
# torque, never publishes /cmd_vel, never opens SDK control sessions, and
# never sends ROS goals.
#
# Usage (from the workstation, or on the robot host):
#   bash scripts/hardware/robot_edge_preflight.sh [ssh_alias]
#
# If an ssh alias (e.g. `rasp_pi`) is given, all checks run over SSH.
set -u

SSH_ALIAS="${1:-}"

run() {
    if [ -n "${SSH_ALIAS}" ]; then
        ssh -o BatchMode=yes "${SSH_ALIAS}" "$*"
    else
        bash -c "$*"
    fi
}

section() { echo; echo "===== $* ====="; }

section "OS / architecture"
run "hostname; uname -m; . /etc/os-release && echo \"\$PRETTY_NAME\"; docker --version"

section "Docker containers (names/images/status)"
run "docker ps -a --format '{{.Names}}\t{{.Image}}\t{{.Status}}'"

section "Docker images (top 12)"
run "docker images --format '{{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.Size}}' | head -12"

section "USB devices"
run "lsusb"

section "LeKiwi serial device"
run "ls -l /dev/lekiwi-base /dev/ttyACM* /dev/ttyUSB* 2>/dev/null; udevadm info --query=property --name=/dev/lekiwi-base 2>/dev/null | grep -E 'ID_SERIAL_SHORT|ID_VENDOR_ID|ID_MODEL_ID'"

section "RealSense camera identity (read-only)"
run "lsusb -v -d 8086:0b3a 2>/dev/null | grep -iE 'iProduct|iSerial' | head -4"

section "CAN interfaces (expected: none in read-only M6)"
run "ip -brief link show type can 2>/dev/null || true"

section "Network (IPs will be redacted in committed reports)"
run "ip -brief addr | grep -v '^lo' | grep UP"

section "Disk / CPU / memory"
run "df -h / | tail -1; nproc; free -h | head -2"

section "udev rules relevant to robot hardware"
run "ls /etc/udev/rules.d/ | grep -iE 'lekiwi|realsense|can|tty'"

section "E-stop / watchdog related (informational, may be absent)"
run "ls /sys/class/gpio/ 2>/dev/null | head -8; ls /etc/systemd/system/ 2>/dev/null | grep -iE 'estop|safety|watchdog' || echo '(none found)'"

echo
echo "Preflight inventory complete (read-only, no motion, no torque enable)."
