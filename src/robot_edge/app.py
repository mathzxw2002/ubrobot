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


# Global runtime and auth (for simplicity in fixture mode)
_runtime: RobotEdgeRuntime | None = None
_token_verifier: TokenVerifier | None = None
_replay_protection: ReplayProtection | None = None


def get_runtime() -> RobotEdgeRuntime:
    """Get the global runtime instance."""
    if _runtime is None:
        raise RuntimeError("Runtime not initialized")
    return _runtime


def get_token_verifier() -> TokenVerifier:
    """Get the global token verifier."""
    if _token_verifier is None:
        raise RuntimeError("Auth not initialized")
    return _token_verifier


def get_replay_protection() -> ReplayProtection:
    """Get the global replay protection."""
    if _replay_protection is None:
        raise RuntimeError("Auth not initialized")
    return _replay_protection


async def get_current_scopes(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    verifier: TokenVerifier = Depends(get_token_verifier),
) -> set[str]:
    """Get current scopes from bearer token.

    Raises:
        HTTPException: If no token or invalid token.
    """
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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan - initialize and clean up runtime and auth."""
    global _runtime, _token_verifier, _replay_protection

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

    _token_verifier = TokenVerifier(auth_config)
    _replay_protection = ReplayProtection(auth_config)

    # Initialize runtime
    execution_mode = getattr(app.state, "execution_mode", "fixture")
    backend = FixtureBackend()
    _runtime = RobotEdgeRuntime(backend=backend)

    yield

    _runtime = None
    _token_verifier = None
    _replay_protection = None


def create_app(
    execution_mode: str = "fixture",
    test_tokens: dict[str, list[str]] | None = None,
) -> FastAPI:
    """Create the Robot Edge FastAPI application."""
    app = FastAPI(
        title="Robot Edge API",
        version=PROTOCOL_VERSION,
        lifespan=lifespan,
    )
    app.state.execution_mode = execution_mode
    if test_tokens is not None:
        app.state.test_tokens = test_tokens

    # Health endpoints (no auth required)
    @app.get("/v1/health/live")
    async def get_health_live() -> dict[str, Any]:
        """Liveness probe."""
        return {"status": "alive"}

    @app.get("/v1/health/ready")
    async def get_health_ready() -> dict[str, Any]:
        """Readiness probe with mode and authority."""
        runtime = get_runtime()
        return {
            "status": "ready",
            "execution_mode": runtime.execution_mode,
            "hardware_authority": runtime.hardware_authority,
        }

    # Observe-scoped endpoints
    @app.get("/v1/capabilities", dependencies=[Depends(require_scope(Scope.OBSERVE.value))])
    async def get_capabilities() -> dict[str, Any]:
        """Get capability inventory."""
        runtime = get_runtime()
        capabilities = runtime.get_capabilities()
        return {
            "capabilities": {
                name.value: snapshot.model_dump(mode="json")
                for name, snapshot in capabilities.items()
            },
        }

    @app.get("/v1/telemetry/snapshot", dependencies=[Depends(require_scope(Scope.OBSERVE.value))])
    async def get_telemetry_snapshot() -> dict[str, Any]:
        """Get telemetry snapshot."""
        runtime = get_runtime()
        telemetry = runtime.get_telemetry_snapshot()
        return {
            "channels": {
                channel.value: snapshot.model_dump(mode="json")
                for channel, snapshot in telemetry.items()
            },
        }

    @app.get("/v1/events", dependencies=[Depends(require_scope(Scope.OBSERVE.value))])
    async def get_events(after: int = 0) -> dict[str, Any]:
        """Get events since the given event ID."""
        runtime = get_runtime()
        events = runtime.get_events_since(after)
        return {
            "events": [event.model_dump(mode="json") for event in events],
        }

    # Command endpoints
    @app.post("/v1/commands", dependencies=[Depends(require_scope(Scope.TASK_SUBMIT.value))])
    async def submit_command(
        request: CommandRequest,
        replay: ReplayProtection = Depends(get_replay_protection),
    ) -> dict[str, Any]:
        """Submit a command for execution."""
        # Check replay protection
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

        runtime = get_runtime()
        command_id = runtime.submit_command(
            text=request.text,
            operator_id=request.operator_id,
            correlation_id=request.correlation_id,
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
    ) -> dict[str, Any]:
        """Cancel an active command."""
        # Check replay protection
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

        runtime = get_runtime()
        cancelled = runtime.cancel_command(
            command_id=command_id,
            operator_id=request.operator_id,
        )

        return {"cancelled": cancelled, "correlation_id": request.correlation_id}

    # Safety endpoint
    @app.post("/v1/safety/stop", dependencies=[Depends(require_scope(Scope.SAFETY_STOP.value))])
    async def emergency_stop(
        request: EmergencyStopRequest,
        replay: ReplayProtection = Depends(get_replay_protection),
    ) -> dict[str, Any]:
        """Trigger emergency stop (bypasses lease requirements)."""
        # Check replay protection
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

        runtime = get_runtime()
        runtime.emergency_stop(
            operator_id=request.operator_id,
            correlation_id=request.correlation_id,
        )

        return {
            "latched": True,
            "correlation_id": request.correlation_id,
        }

    # Lease endpoints
    @app.post("/v1/lease/acquire", dependencies=[Depends(require_scope(Scope.LEASE_MANAGE.value))])
    async def acquire_lease(
        request: LeaseAcquireRequest,
        replay: ReplayProtection = Depends(get_replay_protection),
    ) -> dict[str, Any]:
        """Acquire or renew a navigation lease."""
        # Check replay protection
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

        runtime = get_runtime()
        lease = runtime.acquire_lease(
            operator_id=request.operator_id,
            duration_sec=request.duration_sec,
        )

        return lease.model_dump(mode="json")

    @app.get("/v1/lease", dependencies=[Depends(require_scope(Scope.OBSERVE.value))])
    async def get_lease() -> dict[str, Any] | None:
        """Get current lease, if any."""
        runtime = get_runtime()
        lease = runtime.get_lease()
        return lease.model_dump(mode="json") if lease else None

    return app


if __name__ == "__main__":
    import uvicorn
    app = create_app(execution_mode="fixture")
    uvicorn.run(app, host="127.0.0.1", port=8780)
