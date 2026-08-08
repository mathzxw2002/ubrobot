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
- `UBROBOT_EDGE_ESTOP_ENABLED`: "true" to bind the physical E-stop (M7). Leave unset/`false` means local stop is unavailable and must be reported as such; hardware authority must not be claimed while it is unbound.
- `UBROBOT_EDGE_ESTOP_CHIP`: gpiod chip for the E-stop contact, e.g. `gpiochip4` (Raspberry Pi 5)
- `UBROBOT_EDGE_ESTOP_LINE`: GPIO line number for the NC auxiliary contact
- `UBROBOT_EDGE_ESTOP_LINE_NAME`: diagnostic name, default `ubrobot-estop`
- `UBROBOT_EDGE_ESTOP_DEBOUNCE_SEC`: contact-open debounce window, default 0.02

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

## Physical E-stop binding (M7)

Owner-approved configuration (2026-08-02): NC (normally-closed) mushroom
E-stop; the main contacts cut the LeKiwi motor power through a contactor
(power-off is the final, software-independent layer), and the auxiliary
contact is read by the Pi through libgpiod as the software input.

Wiring: auxiliary contact between 3.3 V and the chosen GPIO line; the line
is configured with internal PULL_DOWN, so a pressed button **or a broken
wire** reads low and triggers the fail-closed stop.

Software chain (wired into the running service since 2026-08-03;
`src/robot_edge/hardware/local_stop.py` + `app.py` lifespan):

```text
GPIO contact  ->  LocalStopButton (debounced, fail-closed)
              ->  runtime.local_emergency_stop()  (latches supervisor,
                  cancels the active command, emits the critical event)
              ->  stop fan-out sinks
              ->  safety.emergency_stop event (priority=critical)
              ->  explicit authorized /v1/safety/reset required
```

Runtime behavior:

- Set `UBROBOT_EDGE_ESTOP_ENABLED=true` (with chip/line) to bind; the
  service aborts startup on missing/invalid config (fail-closed).
- `/v1/health/ready` reports `local_stop.bound`, `source`, and
  `contact_closed` truthfully from the first poll.
- `/v1/safety/reset` re-arms the button: a still-open contact re-latches
  and re-executes the stop fan-out; the reset is never trusted blindly.
- Hardware authority (`UBROBOT_EDGE_HARDWARE_AUTHORITY=true` in hardware
  mode) is refused unless the E-stop is bound.

Robot-side latency measurement (wheels lifted, torque disabled, dry-run
first):

```bash
python3 scripts/hardware/measure_stop_latency.py \
  --chip gpiochip4 --line <PIN> --line-name ubrobot-estop --dry-run
```

`--execute` adds the real fan-out (three zero `/cmd_vel` messages, then
`docker stop` of the driver container, which deactivates ros2_control and
disables torque via SIGINT). Run `--execute` only after the dry-run passes
and the physical E-stop was verified by a human.

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

## Release & versioning (P2)

**Semver tags.** Image tags must be immutable and carry commit identity. Use
`sematic version + date + commit short hash`:

```text
ubrobot/robot-edge:1.2.0-20260808-1a2b3c4d
```

Build a release from a `vX.Y.Z` git tag so the image is reproducible:

```bash
git tag v1.2.0 && git push origin v1.2.0
docker build -f deploy/robot-edge/Dockerfile \
  -t ubrobot/robot-edge:1.2.0-$(date +%Y%m%d)-$(git rev-parse --short HEAD) .
```

Never push to `:latest` or `:hardware` for a real deployment; those are
dev-only convenience aliases.

**Image signing (recommended).** Sign every release image with cosign and
verify in CI so a tampered registry can never inject a build:

```bash
cosign sign --key cosign.key ubrobot/robot-edge:1.2.0-20260808-1a2b3c4d
# verify in deploy pipeline before `docker compose up`:
cosign verify --key cosign.pub ubrobot/robot-edge:1.2.0-20260808-1a2b3c4d
```

Requires a key pair stored in a secret manager; this repo does not commit any
signing keys.

**Observability.** Robot Edge exports Prometheus metrics at `/v1/metrics`
(gauge-style, no auth; non-secret counters only):

- `ubrobot_edge_commands_total{state=...}`
- `ubrobot_edge_lease_active`
- `ubrobot_edge_safety_latched`
- `ubrobot_edge_capability_available{capability=...}`
- `ubrobot_edge_estop_triggered`

Scrape it from a Prometheus instance (or a hosted scraper) configured on the
robot network. `/v1/metrics` returns 503 when `prometheus-client` is not
installed (fixture/dev mode).

**Non-root container.** The `robot-edge` image runs as a dedicated non-root
`ubrobot` user. Do not add `USER root` or `privileged: true` unless the change
is reviewed: the service binds a TCP port and reads a read-only token file, so
it needs no privileges.

## Production stack (2026-08-03, M7 complete)

Production images on the Raspberry Pi (host network, ROS domain 0, UDP-only
Fast DDS):

- `ubrobot/lekiwi-base-driver:0.2.0-rc1-m7-20260803` — torque lifecycle via
  `compose.hardware[-torque-test].yaml` (BEST_EFFORT cmd_vel adapter).
- `ubrobot/emos:jazzy-m7-20260803` — full navigation stack (bringup +
  recipe with Cortex/ARK planner, RoboML detection, Kompass vision stack).
  Built with the kompass float32 timestamp patch, recipe config-mode vision
  setup, cortex tool-args patch, and optional GraspObject import.
- Recipe container must mount the shared Fast DDS profile
  (`deploy/fastdds/udp-only.xml`) or DDS discovery inside it fails
  (CriticalZoneChecker never initializes).

Launch the full chain:

1. bringup: `cortex_navigation_bringup.launch.py start_sensors:=true`
2. recipe: `/opt/ubrobot/recipes/cortex_navigation/recipe.py` (full mode)
   with `CORTEX_MODEL_*` (ARK via planner relay :18081) and
   `ROBOML_HOST=192.168.18.230 ROBOML_PORT=6379` (rtdetr detection).
3. Navigation guard parameters are verified loaded
   (`lease_timeout_sec=0.25`, `raw_command_timeout_sec=0.25`,
   `guard_period_sec=0.05`); the launch-time "parameter not supported"
   warnings come from the RealSense nested launch and are harmless noise.

Hardware motion (M7 Task 13) validated: bounded motion, lease-expiry
fail-closed stop, normal cancel, and NavigateToObject with real Kompass
vision tracking (chair, delta ~0.4 m). Torque remains disabled after
validation.

## Go2+Piper hardware acceptance (Task 6)

The Go2+Piper combined acceptance uses a dedicated deployment gate that
fails closed at startup when any gate is missing. It is the ONLY sanctioned
path to run the Edge with real Go2+Piper hardware authority.

Gates (all required; see `compose.go2-piper.hardware.yaml`):

- `UBROBOT_PLATFORM=go2_piper`, `UBROBOT_GRASP_PLATFORM=go2_piper`
- `UBROBOT_EDGE_MODE=hardware` + `UBROBOT_EDGE_HARDWARE_AUTHORITY=true`
- `UBROBOT_EDGE_ESTOP_ENABLED=true` with `CHIP`/`LINE` (bound local stop)
- `REMOTE_PERCEPTION_SERVICE_URL` (x86 GPU server)
- `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` (Go2 DDS is CycloneDDS, Task 1)
- Reviewed checklist `deploy/robot-edge/checklist/go2-piper-hardware-checklist.md`

Bring-up:

```bash
docker compose -f deploy/robot-edge/compose.fixture.yaml \
               -f deploy/robot-edge/compose.go2-piper.hardware.yaml up
```

Acceptance driver (gate + mutual-exclusion safety, no unsupervised motion):

```bash
python tests/hardware/test_go2_piper_cortex_acceptance.py            # workstation
python tests/hardware/test_go2_piper_cortex_acceptance.py --hardware # operator-driven
```

Stage plan: read-only health -> zero-output/stop -> (low-speed Go2
navigation DEFERRED) -> stationary Piper pre-grasp -> light grasp. Piper
arm stages can run first on the stationary dog; Go2 navigation on the real
dog requires an explicit operator release.

Mutual-exclusion bottom line (codified in the acceptance harness): Grasp
while a navigation lease is active is REJECTED; a navigation lease appearing
mid-grasp cancels the grasp fail-closed; remote-perception unreachable never
produces motion.
