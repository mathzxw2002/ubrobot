"""Robot Edge FastAPI application."""

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.responses import JSONResponse

from ubrobot_contracts import PROTOCOL_VERSION
from ubrobot_contracts.capabilities import CapabilityName
from ubrobot_contracts.edge_api import (
    CommandRequest,
    CommandAccepted,
    CancelRequest,
    EmergencyStopRequest,
    LeaseAcquireRequest,
    LeaseRecord,
    ErrorResponse,
)

from robot_edge.auth import AuthConfig, TokenVerifier, ReplayProtection, Scope
from robot_edge.fixture_backend import FixtureBackend
from robot_edge.runtime import RobotEdgeRuntime


# Security scheme for bearer tokens
security = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan - initialize and clean up runtime and auth on the app."""
    # Load auth config
    tokens_file = os.environ.get("UBROBOT_EDGE_TOKENS_FILE")
    tokens: dict[str, list[str]] = {}
    if tokens_file and os.path.exists(tokens_file):
        import json
        with open(tokens_file) as f:
            tokens = json.load(f)

    # For testing, allow in-memory tokens
    test_tokens = getattr(app.state, "test_tokens", None)
    if test_tokens is not None:
        tokens = test_tokens

    request_max_age = int(os.environ.get("UBROBOT_EDGE_REQUEST_MAX_AGE_SEC", "300"))
    nonce_ttl = int(os.environ.get("UBROBOT_EDGE_NONCE_TTL_SEC", "600"))

    auth_config = AuthConfig(
        tokens=tokens,
        request_max_age_sec=request_max_age,
        nonce_ttl_sec=nonce_ttl,
    )

    app.state.token_verifier = TokenVerifier(auth_config)
    app.state.replay_protection = ReplayProtection(auth_config)

    execution_mode = getattr(app.state, "execution_mode", "fixture")
    # Tests may inject a step delay directly; the environment variable is the
    # deployment path (compose profiles / E2E subprocess). Default zero keeps
    # unit tests fast.
    step_delay = getattr(app.state, "fixture_step_delay_sec", None)
    if step_delay is None:
        step_delay = float(os.environ.get("UBROBOT_EDGE_FIXTURE_STEP_DELAY_SEC", "0.0"))
    app.state.runtime = RobotEdgeRuntime(
        backend=_create_backend(execution_mode, fixture_step_delay_sec=float(step_delay))
    )

    yield

    app.state.runtime = None
    app.state.token_verifier = None
    app.state.replay_protection = None


def _create_backend(execution_mode: str, *, fixture_step_delay_sec: float):
    """Create the runtime backend for the execution mode.

    - ``fixture`` (default): deterministic fixture backend, no hardware.
    - ``hardware``: read-only ROS graph backend (M6). Imports rclpy lazily
      inside the factory; fixture/mock modes never touch the ROS stack.
      ``fixture_step_delay_sec`` widens the fixture active window for
      process-level cancel/E-stop tests (<=100 ms per step).
    """
    if execution_mode == "hardware":
        from robot_edge.ros.backend import create_readonly_ros_backend

        backend = create_readonly_ros_backend(execution_mode="hardware")
        if backend is None:
            raise RuntimeError("hardware mode requested but ROS context unavailable")
        return backend
    return FixtureBackend(step_delay_sec=fixture_step_delay_sec)


def get_runtime(request: Request) -> RobotEdgeRuntime:
    """Dependency: the per-app runtime instance."""
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Robot Edge runtime not initialized",
        )
    return runtime


def get_token_verifier(request: Request) -> TokenVerifier:
    """Dependency: the per-app token verifier."""
    verifier = getattr(request.app.state, "token_verifier", None)
    if verifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Robot Edge auth not initialized",
        )
    return verifier


def get_replay_protection(request: Request) -> ReplayProtection:
    """Dependency: the per-app replay protection."""
    replay = getattr(request.app.state, "replay_protection", None)
    if replay is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Robot Edge auth not initialized",
        )
    return replay


async def get_current_scopes(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    verifier: TokenVerifier = Depends(get_token_verifier),
) -> set[str]:
    """Resolve scopes from the bearer token; 401 on missing/invalid token."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication",
        )
    try:
        return verifier.verify_token(credentials.credentials)
    except PermissionError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication",
        )


def require_scope(required_scope: str):
    """Dependency factory to require a specific scope."""
    async def dependency(scopes: set[str] = Depends(get_current_scopes)) -> set[str]:
        if required_scope not in scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required scope: {required_scope}",
            )
        return scopes
    return dependency


def create_app(
    execution_mode: str = "fixture",
    test_tokens: dict[str, list[str]] | None = None,
    fixture_step_delay_sec: float | None = None,
) -> FastAPI:
    """Create the Robot Edge FastAPI application.

    ``fixture_step_delay_sec`` widens the active-command window in the fixture
    backend so cancellation/E-stop tests can observe a command mid-flight.
    Kept <= 100 ms per the plan's test-time constraint.
    """
    app = FastAPI(
        title="Robot Edge API",
        version=PROTOCOL_VERSION,
        lifespan=lifespan,
    )
    app.state.execution_mode = execution_mode
    app.state.runtime = None
    app.state.token_verifier = None
    app.state.replay_protection = None
    if test_tokens is not None:
        app.state.test_tokens = test_tokens
    if fixture_step_delay_sec is not None:
        app.state.fixture_step_delay_sec = float(fixture_step_delay_sec)

    # Health endpoints (no auth required)
    @app.get("/v1/health/live")
    async def get_health_live() -> dict[str, Any]:
        """Liveness probe."""
        return {"status": "alive"}

    @app.get("/v1/health/ready")
    async def get_health_ready(
        runtime: RobotEdgeRuntime = Depends(get_runtime),
    ) -> dict[str, Any]:
        """Readiness probe with mode and authority."""
        return {
            "status": "ready",
            "execution_mode": runtime.execution_mode,
            "hardware_authority": runtime.hardware_authority,
        }

    # Observe-scoped endpoints
    @app.get("/v1/capabilities", dependencies=[Depends(require_scope(Scope.OBSERVE.value))])
    async def get_capabilities(
        runtime: RobotEdgeRuntime = Depends(get_runtime),
    ) -> dict[str, Any]:
        """Get capability inventory."""
        capabilities = runtime.get_capabilities()
        return {
            "capabilities": {
                name.value: snapshot.model_dump(mode="json")
                for name, snapshot in capabilities.items()
            },
        }

    @app.get("/v1/telemetry/snapshot", dependencies=[Depends(require_scope(Scope.OBSERVE.value))])
    async def get_telemetry_snapshot(
        runtime: RobotEdgeRuntime = Depends(get_runtime),
    ) -> dict[str, Any]:
        """Get telemetry snapshot."""
        telemetry = runtime.get_telemetry_snapshot()
        return {
            "channels": {
                channel.value: snapshot.model_dump(mode="json")
                for channel, snapshot in telemetry.items()
            },
        }

    @app.get("/v1/events", dependencies=[Depends(require_scope(Scope.OBSERVE.value))])
    async def get_events(
        after: int = 0,
        runtime: RobotEdgeRuntime = Depends(get_runtime),
    ) -> dict[str, Any]:
        """Get events since the given event ID (cursor replay)."""
        events = runtime.get_events_since(after)
        return {
            "events": [event.model_dump(mode="json") for event in events],
        }

    # Command endpoints
    @app.post("/v1/commands", dependencies=[Depends(require_scope(Scope.TASK_SUBMIT.value))])
    async def submit_command(
        request: CommandRequest,
        replay: ReplayProtection = Depends(get_replay_protection),
        runtime: RobotEdgeRuntime = Depends(get_runtime),
    ) -> dict[str, Any]:
        """Submit a command for execution."""
        if not replay.check_timestamp(request.timestamp):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Request timestamp out of range",
            )
        if not replay.check_and_store_nonce(request.nonce):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Request nonce already used",
            )

        try:
            command_id = runtime.submit_command(
                text=request.text,
                operator_id=request.operator_id,
                correlation_id=request.correlation_id,
            )
        except RuntimeError as exc:
            # A latched safety state or rejected command is a precondition
            # conflict, not a server error.
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            )

        accepted = CommandAccepted(
            command_id=command_id,
            correlation_id=request.correlation_id,
        )
        return accepted.model_dump(mode="json")

    @app.post("/v1/commands/{command_id}/cancel", dependencies=[Depends(require_scope(Scope.TASK_CANCEL.value))])
    async def cancel_command(
        command_id: str,
        request: CancelRequest,
        replay: ReplayProtection = Depends(get_replay_protection),
        runtime: RobotEdgeRuntime = Depends(get_runtime),
    ) -> dict[str, Any]:
        """Cancel an active command."""
        if not replay.check_timestamp(request.timestamp):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Request timestamp out of range",
            )
        if not replay.check_and_store_nonce(request.nonce):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Request nonce already used",
            )

        cancelled = runtime.cancel_command(
            command_id=command_id,
            operator_id=request.operator_id,
        )
        if not cancelled:
            # Nothing was cancelled: the command is unknown or already
            # terminal. 409 tells the operator that no cancellation took
            # effect instead of a misleading 200 "acknowledged".
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="No active command with that ID",
            )

        return {"cancelled": cancelled, "correlation_id": request.correlation_id}

    # Safety endpoint
    @app.post("/v1/safety/stop", dependencies=[Depends(require_scope(Scope.SAFETY_STOP.value))])
    async def emergency_stop(
        request: EmergencyStopRequest,
        replay: ReplayProtection = Depends(get_replay_protection),
        runtime: RobotEdgeRuntime = Depends(get_runtime),
    ) -> dict[str, Any]:
        """Trigger emergency stop (bypasses lease requirements)."""
        if not replay.check_timestamp(request.timestamp):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Request timestamp out of range",
            )
        if not replay.check_and_store_nonce(request.nonce):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Request nonce already used",
            )

        runtime.emergency_stop(
            operator_id=request.operator_id,
            correlation_id=request.correlation_id,
        )

        return {
            "latched": True,
            "correlation_id": request.correlation_id,
        }

    @app.post("/v1/safety/reset", dependencies=[Depends(require_scope(Scope.SAFETY_STOP.value))])
    async def reset_safety(
        request: EmergencyStopRequest,
        replay: ReplayProtection = Depends(get_replay_protection),
        runtime: RobotEdgeRuntime = Depends(get_runtime),
    ) -> dict[str, Any]:
        """Reset the latched safety state (explicit, authorized reset only).

        Reuses the safety.stop scope: an operator authorized to stop is
        authorized to clear the latch after verifying the scene is safe.
        """
        if not replay.check_timestamp(request.timestamp):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Request timestamp out of range",
            )
        if not replay.check_and_store_nonce(request.nonce):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Request nonce already used",
            )

        runtime.reset_safety(operator_id=request.operator_id, authorized=True)
        return {
            "latched": runtime.safety_latched,
            "correlation_id": request.correlation_id,
        }

    # Lease endpoints
    @app.post("/v1/lease/acquire", dependencies=[Depends(require_scope(Scope.LEASE_MANAGE.value))])
    async def acquire_lease(
        request: LeaseAcquireRequest,
        replay: ReplayProtection = Depends(get_replay_protection),
        runtime: RobotEdgeRuntime = Depends(get_runtime),
    ) -> dict[str, Any]:
        """Acquire or renew a navigation lease."""
        if not replay.check_timestamp(request.timestamp):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Request timestamp out of range",
            )
        if not replay.check_and_store_nonce(request.nonce):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Request nonce already used",
            )

        lease = runtime.acquire_lease(
            operator_id=request.operator_id,
            duration_sec=request.duration_sec,
        )

        return lease.model_dump(mode="json")

    @app.get("/v1/lease", dependencies=[Depends(require_scope(Scope.OBSERVE.value))])
    async def get_lease(
        runtime: RobotEdgeRuntime = Depends(get_runtime),
    ) -> dict[str, Any] | None:
        """Get current lease, if any."""
        lease = runtime.get_lease()
        return lease.model_dump(mode="json") if lease else None

    return app


if __name__ == "__main__":
    import uvicorn

    execution_mode = os.environ.get("UBROBOT_EDGE_MODE", "fixture")
    app = create_app(execution_mode=execution_mode)
    host = os.environ.get("UBROBOT_EDGE_HOST", "127.0.0.1")
    port = int(os.environ.get("UBROBOT_EDGE_PORT", "8780"))
    log_level = os.environ.get("UBROBOT_EDGE_LOG_LEVEL", "info")
    uvicorn.run(app, host=host, port=port, log_level=log_level)
