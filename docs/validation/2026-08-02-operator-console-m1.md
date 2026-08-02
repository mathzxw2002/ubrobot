# Operator Console M1 validation

Date: 2026-08-02  
Environment: Windows workstation, hardware disconnected  
Scope: dependency baseline, process lifecycle, health, shutdown; mock only

## Result

M1 passed. No Raspberry Pi, Piper, Go2, RealSense, ROS hardware driver, motor,
or navigation actuator was initialized.

## Dependency baseline

The hardware-free Windows runtime is defined by
`requirements-operator-console.txt`; test tooling is defined by
`requirements-dev.txt`. The full `requirements.txt` remains the Linux/robot/ML
lock and is not used to bootstrap the workstation console.

Command:

```powershell
python -m pip install --dry-run -r requirements-operator-console.txt
```

Result: dependency resolution completed successfully for the current Windows
Python environment. The active global environment still contains unrelated
historical `lerobot`/packaging/setuptools drift; it does not affect the pinned
Operator Console subset.

## Automated validation

Command:

```powershell
python -m unittest discover -s tests/cortex_navigation -p "test_*.py"
```

Result: 115 tests passed in 4.204 seconds.

Coverage added in M1:

- supported FastAPI/Gradio/Starlette/Uvicorn/websockets versions;
- mock FastAPI app construction without media or hardware;
- free and occupied port diagnosis, including owning PID;
- liveness and sanitized readiness responses;
- local token authorization for graceful shutdown;
- voice-session and backend cleanup during FastAPI shutdown.

## Real process validation

An isolated mock instance was exercised on port 17863:

```powershell
./scripts/operator_console.ps1 start -Port 17863
./scripts/operator_console.ps1 status -Port 17863
./scripts/operator_console.ps1 start -Port 17863
./scripts/operator_console.ps1 stop -Port 17863
./scripts/operator_console.ps1 status -Port 17863
```

Observed results:

1. The first start reached `/api/health/ready` with
   `backend=cortex-mock`, `voice=disabled`, and `mode=mock`.
2. The duplicate start was rejected and identified owner PID 40200.
3. Stop used the loopback-only token endpoint; Uvicorn ran application
   shutdown, VoiceSessionManager stopped, and the backend close hook ran.
4. PID 40200 exited and port 17863 was free after shutdown.

The process logs were written to `logs/operator-console-17863.stdout.log` and
`logs/operator-console-17863.stderr.log`. Runtime PID/token files were removed
by the launcher after shutdown.

## Known non-blocking output

Gradio 5.50 emits forward-looking Gradio 6 deprecation warnings for mounted
`head`/event API parameters, plus test-only asyncio ResourceWarnings from
Gradio's internal client/version-check loops. The supported version is pinned
to 5.50.0 for M1; migration of browser event delivery is scheduled for M2,
where the polling/embedded-script integration is replaced by the operator
event stream.
