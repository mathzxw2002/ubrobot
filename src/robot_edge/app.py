"""Robot Edge FastAPI application."""

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from ubrobot_contracts import PROTOCOL_VERSION
from ubrobot_contracts.capabilities import CapabilityName
from ubrobot_contracts.edge_api import (
    HealthResponse,
)

from robot_edge.fixture_backend import FixtureBackend
from robot_edge.runtime import RobotEdgeRuntime


# Global runtime (for simplicity in fixture mode)
_runtime: RobotEdgeRuntime | None = None


def get_runtime() -> RobotEdgeRuntime:
    """Get the global runtime instance."""
    if _runtime is None:
        raise RuntimeError("Runtime not initialized")
    return _runtime


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan - initialize and clean up runtime."""
    global _runtime
    execution_mode = getattr(app.state, "execution_mode", "fixture")
    backend = FixtureBackend()
    _runtime = RobotEdgeRuntime(backend=backend)
    yield
    _runtime = None


def create_app(execution_mode: str = "fixture") -> FastAPI:
    """Create the Robot Edge FastAPI application."""
    app = FastAPI(
        title="Robot Edge API",
        version=PROTOCOL_VERSION,
        lifespan=lifespan,
    )
    app.state.execution_mode = execution_mode

    # Health endpoints
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

    # Capabilities endpoint
    @app.get("/v1/capabilities")
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

    # Telemetry endpoint
    @app.get("/v1/telemetry/snapshot")
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

    # Command endpoints (stubbed for fixture mode without auth)
    @app.post("/v1/commands")
    async def submit_command(request: Request) -> dict[str, Any]:
        """Submit a command (stub - requires auth in real implementation)."""
        # In real implementation, this would validate auth and submit
        raise HTTPException(status_code=501, detail="Not implemented in fixture mode")

    @app.post("/v1/commands/{command_id}/cancel")
    async def cancel_command(command_id: str) -> dict[str, Any]:
        """Cancel a command (stub)."""
        raise HTTPException(status_code=501, detail="Not implemented in fixture mode")

    @app.post("/v1/safety/stop")
    async def emergency_stop() -> dict[str, Any]:
        """Trigger emergency stop (stub)."""
        raise HTTPException(status_code=501, detail="Not implemented in fixture mode")

    # Event stream (stub)
    @app.get("/v1/events")
    async def get_events(after: int = 0) -> dict[str, Any]:
        """Get events since the given event ID (stub)."""
        runtime = get_runtime()
        events = runtime.get_events_since(after)
        return {
            "events": [event.model_dump(mode="json") for event in events],
        }

    return app


if __name__ == "__main__":
    import uvicorn
    app = create_app(execution_mode="fixture")
    uvicorn.run(app, host="127.0.0.1", port=8780)
