# Robot Edge Fixture Validation Report (M5)

- Date/time: 2026-08-02 20:07 (+08:00)
- Commit: `d904965` (M5 tasks 1–5) + uncommitted worktree for tasks 6–8, validated before the final M5 commits
- Machine role: workstation (developer PC, Windows)
- OS/architecture: Windows (AMD64), Python 3.13.2 (Anaconda)
- Execution mode: `cortex-mock` (console) + `fixture` (Robot Edge)
- Mobile-base profile: none selected yet (M6 inventory)
- Hardware authority: **false** everywhere
- Safety controls present in fixture: lease expiry fail-closed, safety latch, emergency-stop endpoint, authorized reset
- Hardware/ROS/cloud tests: **not executed** (hardware disconnected by design in M5)

## Scope

Milestone M5 (fixture-only Robot Edge) acceptance:

1. Versioned shared transport contracts (`ubrobot_contracts`).
2. Standalone Robot Edge service in fixture mode with auth, scopes, replay
   protection, navigation lease, and local safety supervision.
3. Operator Console `robot-edge` backend adapter and telemetry bridge.
4. Two-process fixture acceptance (independent Console + Edge processes).

## Commands

```powershell
PYTHONPATH=src python -m unittest discover -s tests/robot_edge -p "test_*.py" -q
PYTHONPATH=src python -m unittest discover -s tests/cortex_navigation -p "test_*.py" -q
PYTHONPATH=src python -m unittest tests.e2e.test_operator_console_mock -v
PYTHONPATH=src python -m unittest tests.e2e.test_operator_robot_edge_fixture -v
powershell -ExecutionPolicy Bypass -File scripts/validate_operator_console.ps1
```

One-command validation report: `logs/validation/operator-console-software-20260802-200713.md`.

## Results

| Suite | Tests | Result |
|---|---:|---|
| `tests/cortex_navigation` (incl. 15 new robot-edge adapter/telemetry tests) | 165 | PASS |
| `tests/robot_edge` (contracts/auth/lease/safety/fixture service) | 73 | PASS |
| `tests/e2e/test_operator_console_mock` | 3 | PASS |
| `tests/e2e/test_operator_robot_edge_fixture` (two-process) | 8 | PASS |

Two-process fixture E2E exercises, in order: authentication and
`hardware_authority=false`, lease acquire, command submit with ordered
feedback through the Operator API, Edge event timeline replay, status query
without dispatching a second command, mid-flight cancel, emergency stop that
latches and blocks new work, authorized reset, both event streams
reconnecting, and no residual listeners after shutdown.

## Fixes landed during M5 wrap-up

1. **Edge cancel semantics**: `POST /v1/commands/{id}/cancel` now returns 409
   when no active command exists, so an acknowledgement always means a real
   cancellation took effect (previously returned 200 with `cancelled: false`,
   which made `cancel_active()` lie).
2. **Adapter event filtering**: `RobotEdgeBackend._poll_events` filters the
   shared Edge stream by `command_id`; the cursor used to replay from the
   beginning and an earlier command's terminal event could end a new
   `execute()` early.
3. **Deterministic cancel tests**: `create_app(..., fixture_step_delay_sec)`
   widens the fixture active-command window (<=100 ms per step) so
   cancellation is observed mid-flight instead of racing completion.
4. **Test harness pipe-block fix**: the two-process E2E redirected child
   stderr to a PIPE nobody read; once the pipe buffer filled, the Console's
   main thread blocked forever on a log write and the service stopped
   responding. stderr now goes to files (`console.stderr.log`,
   `edge.stderr.log`), which also preserves logs for diagnosis.
5. **One-command validation**: `scripts/validate_operator_console.ps1` sets
   `PYTHONPATH=src` (src-layout package) and includes the two-process
   robot-edge fixture suite; the report records all four suites.

## Known limitations / deferred

- Fixture behavior is not hardware evidence. Motion, sensors, E-stop
  latency, and lease fail-closed behavior on real hardware are M6/M7 scope.
- The Edge fixture uses a `StepSink`-style recorded stop fan-out; no motor or
  ROS function exists in fixture mode.
- Cloud ASR credentials were not present; voice provider is `mock`.
- No mobile-base profile selected; M6 inventory is blocked until the
  robot-side computer, RealSense, base, and Piper are available.

## Ownership review

Per the plan's mandatory checkpoint, M5 stops here for owner review before
any M6 hardware-authority work begins.
