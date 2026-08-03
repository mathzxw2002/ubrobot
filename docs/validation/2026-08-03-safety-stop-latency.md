# Robot Edge Physical E-stop Binding Validation Report (M7, software level)

- Date/time: 2026-08-03 08:20 (+08:00)
- Commit: `0db82a0` (runtime wiring + tests + deployment) on top of
  `d7760a3` (M7 Task 12 building blocks); this report adds one doc commit
- Machine role: workstation (developer PC, Windows, AMD64, Python 3.13.2)
- Execution mode: software fixtures + injected fake E-stop readers; the
  robot-side host (Raspberry Pi) is **not started** in this validation
- Mobile-base profile: **`lekiwi`** (owner-selected, unchanged)
- Hardware authority: **false** everywhere; the M7 authority gate is
  verified at the application level (see below)
- Physical E-stop wiring on robot host: **not yet verified by a human**;
  live latency measurement is deferred per plan Task 12 steps 5–6

## Scope

Milestone M7, Task 12 — "Bind and measure local safety controls", software
portion. This report covers:

1. Wiring the physical E-stop contact into the Robot Edge runtime so a press
   (or broken wire) **latches the safety supervisor, cancels the active
   command, and emits the critical `safety` event** — previously the binding
   was library code only and had no effect on the running service.
2. Fail-closed startup semantics: misconfigured or unreadable E-stop binding
   aborts startup; hardware authority cannot be claimed while the E-stop is
   unbound.
3. The re-arm protocol: after an explicit authorized `/v1/safety/reset`, a
   still-open contact re-latches and re-executes the stop fan-out instead of
   trusting the reset.
4. Read-only diagnostics: `/v1/health/ready` reports `local_stop.bound`,
   `source`, and `contact_closed` with no secrets or descriptors.

Deferred to live hardware (plan Task 12 steps 5–6, blocked per inventory
gate): measured input-detection / supervisor-dispatch / driver-acknowledge /
physical power-off latency segments, and the `--execute` fan-out (zero
`/cmd_vel` ×3 + driver container SIGINT).

## Commands

```powershell
PYTHONPATH=src python -m unittest discover -s tests/robot_edge -p "test_*.py" -q
PYTHONPATH=src python -m unittest discover -s tests/cortex_navigation -p "test_*.py" -q
PYTHONPATH=src python -m unittest tests.e2e.test_operator_robot_edge_fixture -q
```

## Results

| Suite | Tests | Result |
|---|---:|---|
| `tests/robot_edge` (123; **+15 new M7 wiring tests**) | 123 | PASS |
| `tests/cortex_navigation` (regression guard) | 165 | PASS |
| `tests/e2e/test_operator_robot_edge_fixture` (two-process) | 8 | PASS |

## What the new M7 tests assert

### Import boundary

- Importing `robot_edge.app` and `robot_edge.hardware.local_stop` never
  imports `gpiod` (robot-side only); constructing `GpiodEstopLineReader`
  imports it lazily inside `__init__` and fails naturally without libgpiod.
- `gpiod` is added to `requirements-robot-edge.txt` and installed in
  `Dockerfile.ros` (Ubuntu Noble `libgpiod2`), **not** to any workstation
  dependency file.

### Runtime wiring (`app.py` lifespan)

- E-stop disabled (default): `local_stop.bound=false` in readiness; no
  poller thread, no reader, no gpiod import.
- E-stop enabled with a closed contact: poller thread running, readiness
  reports `bound=true`, source, and `contact_closed=true`; the readiness
  body contains no token strings.
- E-stop enabled but chip/line missing, line non-integer, or the reader
  factory returning a non-`EstopLineReader`: **startup aborts** (fail-closed;
  the service refuses to run unprotected).
- One synchronous seed poll at binding time so readiness reflects the
  contact truth immediately, before the first background poll.

### Physical press semantics

- Open contact past the debounce window → `SafetySupervisor` latched,
  active command cancelled, `safety.emergency_stop`-style event emitted with
  `source=local`, `critical=true`.
- Latched state rejects new commands (`409`); explicit authorized
  `/v1/safety/reset` clears the latch; commands accepted again after reset.
- **Re-arm correctness**: after reset, a still-open contact re-latches on
  the next polls, and the stop fan-out is re-executed (supervisor
  `_stop_executed` is cleared by the authorized reset; idempotence within
  one latch cycle is preserved by the latch early-return).
- Lifespan shutdown joins the poller thread and releases the reader.

### Hardware-authority gate (M7)

- `execution_mode=hardware` + `UBROBOT_EDGE_HARDWARE_AUTHORITY=true` without
  `UBROBOT_EDGE_ESTOP_ENABLED=true` → startup aborts with a fail-closed
  `RuntimeError` before any backend is created. Read-only hardware mode
  (authority=false) is unaffected.

## Deployment artifacts

- `requirements-robot-edge.txt`: `gpiod>=2.0` (robot-side containers only).
- `deploy/robot-edge/Dockerfile.ros`: `apt-get install libgpiod2` (Noble).
- `deploy/robot-edge/compose.ros-readonly.yaml`: maps `/dev/gpiochip0..4`
  (Raspberry Pi 5) and exposes `UBROBOT_EDGE_ESTOP_*` env vars, default
  `UBROBOT_EDGE_ESTOP_ENABLED=false`.

## Hardware authority state

- `hardware_authority=false` in every module, snapshot, and test.
- Command/cancel/stop authority is structurally disabled in the read-only
  backend; the authority gate above prevents any future authority=true
  deployment from running with an unbound physical E-stop.

## Known limitations and deferred live validation

1. This report is **software-only evidence**. The physical contact, GPIO
   wiring, and contactor are not present in this environment.
2. Live latency segments (T1 input detection → T2 supervisor dispatch → T3
   driver acknowledgement → T4 physical power-off) are **not measured**.
   `scripts/hardware/measure_stop_latency.py` is ready; the dry-run requires
   the owner to wire the NC auxiliary contact to a GPIO line and run it on
   the Raspberry Pi with wheels lifted and torque disabled.
3. `--execute` mode (zero `/cmd_vel` ×3 + driver container SIGINT) requires
   the supervised lifted-wheel preflight and human verification of the
   physical E-stop; **it is not authorized by this report**.
4. LeKiwi driver container remains stopped in its torque-enabled hard-gate
   configuration (inventory report, critical finding 1). The base's local
   stop primitive binding (plan Task 12 step 2) stays pending.
5. Piper stop/disable binding is deferred to M8 (no CAN on this host).

## Acceptance (Task 12, software portion)

- [x] Physical E-stop state bound to `SafetySupervisor` through the running
      service (latch + active-command cancel + critical event).
- [x] Fail-closed startup on misconfiguration; authority gate verified.
- [x] Safety latch/reset behavior documented and tested, including re-arm
      after reset with a still-open contact.
- [x] Stop fan-out re-execution after authorized reset (sinks re-armed).
- [ ] Live dry-run latency measurement — **deferred**: needs owner wiring
      of the NC auxiliary contact + human verification on the Raspberry Pi.
- [ ] `--execute` fan-out measurement — **blocked** by the torque-enabled
      LeKiwi container gate and lifted-wheel preflight.

Mock/fixture evidence is not hardware evidence. M7 Task 12 is accepted at
the software level only; live latency measurement and M7 Task 13
(navigation validation) stay blocked per the plan's stop conditions until
the owner wires and verifies the physical E-stop and a supervised preflight
is approved.

## Evidence

- Implementation commit: `0db82a0` (feat: wire physical e-stop into robot
  edge runtime (M7)); this report is the follow-up docs commit
- Worktree state: clean after commit
- Repro: the test commands above, plus
  `bash scripts/hardware/robot_edge_preflight.sh rasp_pi` for the next
  on-device step.
