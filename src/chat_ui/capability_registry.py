"""Transport-neutral capability inventory for the Operator Console.

Descriptors contain state only. They never retain ROS clients, SDK handles,
camera objects, robot managers, or executable callbacks.

For shared contracts between Operator Console and Robot Edge, see ubrobot_contracts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import threading
from typing import Iterable

# Re-export from shared contracts for backward compatibility
from ubrobot_contracts.capabilities import (
    CapabilityAvailability,
    CapabilityHealth,
    ExecutionMode,
)


@dataclass(frozen=True)
class CapabilityDescriptor:
    name: str
    availability: CapabilityAvailability
    health: CapabilityHealth
    execution_mode: ExecutionMode
    required_resources: tuple[str, ...]
    hardware_authority: bool = False
    detail: str = ""
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        if not self.name.strip():
            raise ValueError("capability name must be non-empty")
        if self.hardware_authority and self.execution_mode in {
            ExecutionMode.MOCK,
            ExecutionMode.FIXTURE,
        }:
            raise ValueError("mock/fixture capabilities cannot have hardware authority")
        if any(not resource.strip() for resource in self.required_resources):
            raise ValueError("required resources must be non-empty")

    def to_dict(self) -> dict:
        value = asdict(self)
        value["availability"] = self.availability.value
        value["health"] = self.health.value
        value["execution_mode"] = self.execution_mode.value
        value["required_resources"] = list(self.required_resources)
        value["updated_at"] = self.updated_at.isoformat()
        return value


class CapabilityRegistry:
    """Thread-safe registry that exposes immutable serialized descriptors."""

    def __init__(self, descriptors: Iterable[CapabilityDescriptor] = ()):
        self._lock = threading.RLock()
        self._descriptors: dict[str, CapabilityDescriptor] = {}
        for descriptor in descriptors:
            self.register(descriptor)

    def register(self, descriptor: CapabilityDescriptor) -> None:
        with self._lock:
            if descriptor.name in self._descriptors:
                raise ValueError(f"capability already registered: {descriptor.name}")
            self._descriptors[descriptor.name] = descriptor

    def update(
        self,
        name: str,
        *,
        availability: CapabilityAvailability | None = None,
        health: CapabilityHealth | None = None,
        detail: str | None = None,
        hardware_authority: bool | None = None,
    ) -> CapabilityDescriptor:
        with self._lock:
            current = self._descriptors.get(name)
            if current is None:
                raise KeyError(f"unknown capability: {name}")
            updated = replace(
                current,
                availability=availability or current.availability,
                health=health or current.health,
                detail=current.detail if detail is None else str(detail),
                hardware_authority=(
                    current.hardware_authority
                    if hardware_authority is None
                    else bool(hardware_authority)
                ),
                updated_at=datetime.now(timezone.utc),
            )
            self._descriptors[name] = updated
            return updated

    def get(self, name: str) -> CapabilityDescriptor | None:
        with self._lock:
            return self._descriptors.get(name)

    def snapshot(self) -> dict[str, dict]:
        with self._lock:
            return {
                name: descriptor.to_dict()
                for name, descriptor in sorted(self._descriptors.items())
            }


_RESOURCES = {
    "navigation": ("camera", "depth", "odometry", "navigation_lease"),
    "grasp": ("camera", "depth", "joint_states"),
    "observation": ("camera", "depth"),
    "follow": ("camera", "odometry", "navigation_lease"),
    "stop": ("safety_control",),
}


def create_default_registry(
    *,
    execution_mode: ExecutionMode,
    simulated_capabilities: Iterable[str] = (),
) -> CapabilityRegistry:
    simulated = set(simulated_capabilities)
    descriptors = []
    for name, resources in _RESOURCES.items():
        available = name in simulated
        descriptors.append(
            CapabilityDescriptor(
                name=name,
                availability=(
                    CapabilityAvailability.AVAILABLE
                    if available
                    else CapabilityAvailability.DISCONNECTED
                ),
                health=(
                    CapabilityHealth.HEALTHY if available else CapabilityHealth.UNKNOWN
                ),
                execution_mode=execution_mode,
                required_resources=resources,
                hardware_authority=False,
                detail=(
                    "software simulation only"
                    if available
                    else "no fixture or robot-edge connection"
                ),
                updated_at=datetime.now(timezone.utc),
            )
        )
    return CapabilityRegistry(descriptors)
