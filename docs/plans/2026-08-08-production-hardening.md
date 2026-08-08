# Production Hardening Plan (P0: engineering infrastructure)

> **Status:** P0.1 (CI), P0.2 (ruff) and P0.3 (secret scan) implemented 2026-08-08.
> Remaining P1–P3 phases are future work; this document records decisions so
> they survive context switches.

## Objective

Move UBRobot from an experimental workspace toward a production-grade codebase.
The first barrier is engineering infrastructure (P0): automatic gates that make
regressions fail at merge time instead of on the next hardware run.

## P0.1 — CI pipeline (DONE)

`.github/workflows/ci.yml` with three jobs:

- **lint:** ruff `check` + `format --check` on first-party production packages
  (`src/chat_ui`, `src/ubrobot_contracts`, `src/robot_edge`), pinned to
  `ruff==0.14.13` (the repo's locked version). `src/ubrobot` + `src/service`
  remain informational until cleaned.
- **test:** `unittest` matrix on Python 3.10/3.12, installing from
  `requirements-operator-console.txt` + `pillow` (locked versions). Covers
  robot-edge (210), cortex-navigation (202), e2e (11), secrets (4).
- **secrets:** gitleaks working-tree scan + Python static scan.

### Key engineering decisions (validated)

1. **Pin every dependency.** Bare `pip install gradio` pulls starlette 0.52 and
   breaks the runtime-version contract test (11 failures); the pinned
   `requirements-operator-console.txt` gives 202/202 green. Never install
   unpinned in CI.
2. **Pin ruff.** Newer ruff enables new rules (I001, EXE002) by default and
   drifts the gate. Lock `ruff==0.14.13` and bump deliberately.
3. **`test_motion_arbitration` is `continue-on-error`.** It imports
   `ubrobot_manipulation.authority` (a ROS ament package) from the pure-Python
   workstation path. Fix is refactor plan Task 1 (move `AuthorityTracker` into
   `ubrobot_contracts`); tracked in
   `docs/plans/2026-08-08-refactor-experimental-code.md`.
4. **`test_occupied_port...` fails inside Docker** (psutil cannot see the
   container's own PID in `net_connections`); it passes on real runners.

## P0.2 — Ruff configuration (DONE)

`[tool.ruff]` in `pyproject.toml`:

- `exclude`: vendored trees (`ascam_ros2_ws`, `logoplanner`, `third_party`) that
  keep upstream formatting.
- `select = ["E", "F", "I"]`, `ignore = ["E501"]` — line width is owned by
  `ruff format`, not a separate check (avoids conflict).
- `target-version = "py310"`.

Cleaned: `ubrobot_contracts` (1→0), `robot_edge` (17→0), `chat_ui` (~30→0).
Real bugs caught: undefined `ret` in `camera_util.py`, undefined
`PointCloudPerception` in `http_reasoning_server.py`. Dead code removed:
unused `ffmpeg_command` in `chat_ui/utils.py`. `src/ubrobot` + `src/service`
carry 82 legacy violations in experimental code — cleaned in refactor plan
Tasks 2/4.

## P0.3 — Secret scan gate (DONE)

`gitleaks.toml` + CI `secrets` job (working-tree scan via
`zricethezav/gitleaks` Docker image; the `gitleaks-action` requires a
commercial license).

Allowlisted: local dev TLS credentials (`assets/key.pem`/`cert.pem`,
gitignored), the regression test that embeds retired keys, and vendored trees
whose "secrets" are upstream fixtures.

`tests/security/test_hardcoded_secrets.py` guards two retired DashScope keys.

### OUTSTANDING DECISION — git history cleanup (OWNER)

The git **history** contains retired DashScope keys that full-history gitleaks
still flags:

- `sk-78b8ea9b14b944d0a2240408b8c766dd` — re-introduced by `ec3ee21`
  ("Resolve robot code merge conflicts"); key retired by owner 2026-08-08.
- `sk-479fdd23120c4201bff35a107883c7c3` — old startup scripts; removed in
  `ca2111f`, retired.

CI scans the **working tree only** so new commits are guarded without being
blocked by old history. Cleaning history requires:

```bash
# owner-run, AFTER confirming all clones can be re-synced
pip install git-filter-repo
git filter-repo --replace-text secrets.txt   # or --invert-paths per file
git push --force --all
```

Impact: rewrites all commit SHAs; every local clone must be re-cloned or
`git pull --rebase` with care; co-owners must re-sync. For a self-signed dev
repo this is usually worth it only when the repo becomes shared/published.

## Remaining phases

### P1 — code robustness (DONE 2026-08-08: logging + exceptions + mypy + settings)

DONE:
- **Structured logging:** all 30 `print()` in `chat_ui` (pipeline/utils/adapters)
  replaced with `logging`; `robot_edge`/`ubrobot_contracts` had none. Production
  packages now have zero `print()`.
- **Exception discipline:** every bare `except Exception` reviewed. `cancel`/
  `close`/`shutdown` best-effort paths now log at debug with `exc_info=True`;
  fail-closed paths (return False / continue) documented with comments. Added
  module loggers to runtime.py, ros/backend.py, ros/frames.py, robot_edge.py,
  robot_edge_telemetry.py.
- **mypy:** `[tool.mypy]` config (py310, ignore_missing_imports for ROS/hardware
  SDKs). `ubrobot_contracts` + `robot_edge` + `chat_ui` = 45 files clean.
  Fixed real bugs found: `StopSink` self-inheritance renamed to
  `RecordingStopSink`; `mobile_base_health` value dict typing; `go2_health`
  `_age_is_fresh` object arg; `app.py` nullable `chat_pipeline`.

NOT DONE (deferred — risky, touches 40+ env reads across two processes):
- **Centralized settings (pydantic-settings):** `os.environ.get("UBROBOT_*")`
  scattered across `chat_ui` (20+) and `robot_edge` (20+). Doing this
  incrementally (robot_edge first, then chat_ui) as a separate follow-up to
  avoid a large risky one-shot migration.

### P1.4 — centralized settings (DONE 2026-08-08)

`src/ubrobot_contracts/settings.py`:

- `ConsoleSettings` — `UBROBOT_CHAT_*` / `UBROBOT_VOICE_*` / `UBROBOT_QWEN_*` /
  `UBROBOT_MOCK_*` / console-side `UBROBOT_EDGE_*` + `DASHSCOPE_*`, with
  `validation_alias` for the shared-namespace vars (edge/dashscope/mock/qwen
  do not carry the `UBROBOT_CHAT_` prefix).
- `EdgeSettings` — `UBROBOT_EDGE_*` + `UBROBOT_PLATFORM` (alias), with
  Literal/range validation.
- `console_settings()` / `edge_settings()` lru-cached accessors.

Migrated consumers (no more `os.environ.get("UBROBOT_*")`):
- `robot_edge/app.py` — mode/host/port/log_level/platform, estop chip/line/
  debounce, tokens_file, request/nonce TTLs, fixture step delay, authority/
  exempt flags.
- `chat_ui/pipeline.py` — backend selection, mock timings, robot-edge backend
  URL/token/operator, edge telemetry URL/token/hardware-permitted, voice
  provider. `ChatPipeline.__init__` accepts an optional `settings` for tests.
- `chat_ui/app.py` — log level, media, host/port/tls/backend in `__main__`.
- `chat_ui/qwen_realtime.py` — `QwenRealtimeConfig.from_env()` reads
  ConsoleSettings.

Intentionally kept as env reads: `UBROBOT_SHUTDOWN_TOKEN` (runtime-generated
process secret) and `RobotEdgeBackend`'s token fallback (explicit-arg-first).
`tests/robot_edge/test_settings.py` (8 tests) covers defaults, prefix mapping,
and validation. ConsoleSettings/EdgeSettings are pure Python, unit-testable
without ROS/hardware.

### P2 — observability & deployment (DONE 2026-08-08: metrics + non-root + release docs)

- **Prometheus metrics:** `robot_edge/metrics.py` (`EdgeMetrics` with lazy
  `prometheus_client` import, graceful degradation) + `/v1/metrics` endpoint
  (503 when client absent). Gauges: commands_total (by state), lease_active,
  safety_latched, capability_available, estop_triggered. Wired into
  `/v1/health/ready` (lease/safety/estop) and `/v1/capabilities`
  (per-capability gauge). `prometheus-client>=0.20.0` added to
  requirements-robot-edge.txt (optional). Tests: `tests/robot_edge/test_metrics.py`.
- **Non-root container:** robot-edge Dockerfile creates a dedicated `ubrobot`
  user and runs the service as non-root.
- **Release & versioning:** deploy/robot-edge README gained a
  "Release & versioning (P2)" section: semver+date+sha image tags, cosign
  signing/verification workflow, metrics scrape notes, non-root policy.
- **Image signing:** documented (cosign) as a deployment-side practice; no keys
  committed. Actual signing requires a secret-managed key pair + registry —
  owner-setup, not code.

### P3 — test depth (DONE 2026-08-08)

- **Coverage gate (P3.1):** `[tool.coverage.run]` (core pure-Python only, ROS
  backend excluded — rclpy/camera cannot run on workstations). CI runs
  `coverage run` + `--fail-under=80`; current core coverage 86%. Added
  `coverage[toml]==7.15.4` to requirements-dev + CI install.
- **Fault injection (P3.2):** `tests/robot_edge/test_fault_injection.py`
  (8 tests): clock rollback (future timestamp rejected, 30s skew allowed,
  naive timestamp defaults to UTC), malformed/oversized payloads (422),
  missing fields (422), insufficient scope (403), backend non-2xx reports
  FAILED not fake success. Network-partition and stale-telemetry were already
  covered (test_robot_edge_telemetry).
- **Hardware/fixture isolation (P3.3):** new CI `hardware-contract` job runs
  the Go2+Piper workstation safety contract (gate + mutual-exclusion fakes,
  software-only) with `PYTHONPATH=src:ros_depends_ws/src/ubrobot_manipulation`.
  Explicitly NOT hardware acceptance — real hardware still requires the
  `--hardware` driver + physical E-stop + operator. Fixture JSON data is
  exercised by e2e/qwen tests already in CI.

### P2.5 — dependency vulnerability scan (DONE 2026-08-08, with gradio-6 debt)

- CI `dependency-audit` job runs `scripts/ci/dependency_audit.sh` (pip-audit
  against `requirements-operator-console.txt`). Any NEW vulnerability fails;
  known ones are allowlisted with documented reasons.
- Fixed now: `python-dotenv` 1.1.1→1.2.2 (PYSEC-2026-2270), `starlette`
  0.47.2→0.49.1 (PYSEC-2026-1942).
- **OUTSTANDING DEBT (gradio 6 upgrade):** `gradio==5.50.0` pins
  `starlette<1` and `pillow<12`, so these remain vuln (allowlisted):
  - gradio 5.50.0: PYSEC-2026-63..66, 211, 2178-2179 (fixed in gradio 6.x)
  - pillow 11.3.0: PYSEC-2026-165, 2249-2257, 2874, 3451/3453-3454,
    3493-3496 (fixed in pillow >=12.1.1, needs gradio 6)
  - starlette 0.49.1: PYSEC-2026-161, 248-249, 2280-2281 (fixed in 1.x,
    needs gradio 6)
  Upgrading gradio is a major change (fastapi/starlette chain + chat_ui
  compatibility pass for removed APIs like `gr.Chatbot(type=...)`); it is a
  dedicated follow-up, not done here.
- Locking: `requirements-operator-console.txt` is fully pinned (CI + audit
  source of truth). Full `requirements.txt` (torch/CUDA ML stack) is NOT
  pip-compile-able (resolver times out) and stays hand-maintained.

## Stop conditions

- Any change that makes non-legacy paths (cortex/cortex-mock/robot-edge
  backends) depend on hardware or SDKs.
- CI gate that silently excludes real secrets instead of allowlisting
  documented, vendored, or test-only content.
- Ruff/format drift without a deliberate version bump.
