#!/usr/bin/env python3
"""Fail an EMOS image build when the required Cortex API is unavailable."""

from inspect import signature

from agents.components import Cortex
from agents.config import CortexConfig
from agents.ros import Action, Launcher


def require_parameters(symbol: object, expected: set[str]) -> None:
    parameters = set(signature(symbol).parameters)
    missing = expected - parameters
    if missing:
        name = getattr(symbol, "__name__", repr(symbol))
        raise RuntimeError(f"{name} is missing required parameters: {sorted(missing)}")


def main() -> None:
    require_parameters(
        Cortex,
        {"actions", "output", "model_client", "config", "component_name"},
    )
    require_parameters(
        CortexConfig,
        {"max_planning_steps", "max_execution_steps", "monitoring_interval"},
    )
    require_parameters(Action, {"method", "description"})

    # Launcher has no required integration-specific arguments, but importing
    # and inspecting it proves that the recipe-facing orchestration surface is
    # present in this exact image.
    signature(Launcher)

    for symbol in (Cortex, CortexConfig, Action, Launcher):
        print(f"{symbol.__name__}: {signature(symbol)}")


if __name__ == "__main__":
    main()
