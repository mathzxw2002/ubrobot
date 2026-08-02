# Robot Edge Deployment

This directory contains deployment configuration for the Robot Edge service.

## Quick Start - Fixture Mode

1. Create a tokens configuration:

```bash
mkdir -p config
cat > config/tokens.json << 'EOF'
{
  "operator-token-123": ["observe", "task.submit", "task.cancel", "lease.manage"],
  "safety-token-456": ["observe", "safety.stop"],
  "observer-token-789": ["observe"]
}
EOF
```

2. Start the service:

```bash
docker-compose -f compose.fixture.yaml up -d
```

3. Check health:

```bash
curl http://localhost:8780/v1/health/ready
```

## Configuration

### Environment Variables

- `UBROBOT_EDGE_MODE`: "fixture" or "hardware"
- `UBROBOT_EDGE_HARDWARE_AUTHORITY`: "true" or "false" - MUST be true before any physical motion
- `UBROBOT_EDGE_TOKENS_FILE`: Path to tokens JSON file
- `UBROBOT_EDGE_REQUEST_MAX_AGE_SEC`: Max age of request timestamps (default: 300)
- `UBROBOT_EDGE_NONCE_TTL_SEC`: How long to track used nonces (default: 600)
- `UBROBOT_EDGE_SAFETY_CHECKLIST`: Required in hardware mode - path to safety checklist

### Token Format

Tokens are a JSON file mapping token strings to scope lists:

```json
{
  "my-token-here": ["observe", "task.submit", "task.cancel"],
  "another-token": ["observe", "safety.stop", "lease.manage"]
}
```

### Scopes

- `observe`: Read health, capabilities, telemetry, events
- `task.submit`: Submit new commands
- `task.cancel`: Cancel active commands
- `safety.stop`: Trigger emergency stop (bypasses lease)
- `lease.manage`: Acquire and release navigation leases

## Safety Checklist (Hardware Mode)

Before enabling hardware authority, all items must be completed and signed:

- ✅ Physical E-stop button verified functional
- ✅ Robot is in a safe, open area with no people within reachable distance
- ✅ LeKiwi motors are NOT enabled yet
- ✅ RealSense camera is connected but not streaming
- ✅ Go2 is in idle mode
- ✅ Operator console is ready and connected
- ✅ Network latency is acceptable (< 100ms)
- ✅ Lease duration limits are configured
- ✅ Emergency stop scope is assigned to a dedicated, easily accessible token
- ✅ Two-person rule: one operator, one safety observer

## Hardware Mode (M6+)

Hardware mode is blocked in M5. See the full plan for M6 milestones before attempting.

To enable hardware authority (only after M6+):

```bash
# Edit checklist and complete all items
# Then set
export UBROBOT_EDGE_HARDWARE_AUTHORITY=true
docker-compose -f compose.hardware.yaml up -d
```

## Raspberry Pi deployment (M6 read-only inventory)

The robot-side host is the Raspberry Pi (ssh alias `rasp_pi`, Ubuntu 24.04
ARM64, Docker). Inventory is read-only at this stage; see
`docs/validation/2026-08-02-robot-edge-inventory.md` for the recorded state.

Known deployment facts (from history + on-device read-only checks):

- LeKiwi base: `ubrobot/lekiwi-base-driver` container owns the serial device
  `/dev/lekiwi-base` (udev rule `99-lekiwi-base.rules`). The container is
  currently stopped in a torque-enabled hard-gate configuration and must not
  be started until a supervised preflight and a verified physical E-stop
  exist.
- EMOS: `ubrobot/emos` container, host network, privileged, mounts
  `/dev/bus/usb` for the RealSense D435i.
- Both use `rmw_fastrtps_cpp` + `deploy/fastdds/udp-only.xml` (UDP-only Fast
  DDS) with `ROS_DOMAIN_ID=0`.
- No CAN interface is present; Piper is not connected to this host.

Reproduce the read-only inventory with:

```bash
bash scripts/hardware/robot_edge_preflight.sh rasp_pi
```

## Stopping the Service

```bash
docker-compose -f compose.fixture.yaml down
```

## Security Notes

- Never commit real tokens to git
- Use short-lived tokens in production
- Rotate tokens after deployment
- Never expose port 8780 to public networks
- Use HTTPS in production with TLS
