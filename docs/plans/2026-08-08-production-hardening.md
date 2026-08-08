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

### P1 — code robustness (PARTIAL: logging + exceptions + mypy done 2026-08-08)

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

### P2 — observability & deployment (not started)

- **P2:** `/metrics` endpoint, image signing, non-root containers, semver tags,
  config-redaction in all log paths.

### P3 — test depth (not started)

- **P3:** coverage gates, fault-injection tests, hardened fixture/hardware split.

## Stop conditions

- Any change that makes non-legacy paths (cortex/cortex-mock/robot-edge
  backends) depend on hardware or SDKs.
- CI gate that silently excludes real secrets instead of allowlisting
  documented, vendored, or test-only content.
- Ruff/format drift without a deliberate version bump.
