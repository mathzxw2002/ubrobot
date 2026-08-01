# Offline Grasp Fixture Validation

## Scope

This validation is intentionally hardware-free. It does not connect to the
PI, Piper, Unitree Go2, LeKiwi, RealSense, USB, CAN, or a physical ROS
environment.

## Implemented path

```text
Cortex semantic grasp capability
  -> GraspObject lifecycle coordinator
  -> deterministic grasp fixture
```

The fixture is enabled only through the explicit launch configuration
`start_grasp_server=true` with `grasp_executor=fixture`. Real Piper and Go2
executors remain unavailable and fail closed.

## Assertions

- approach, align, grasp, and retreat feedback is emitted;
- navigation lease rejects a grasp before motion starts;
- a navigation lease appearing during grasp cancels the fixture;
- outer cancellation is acknowledged within the bounded timeout;
- the fixture contains no ROS, torch, SDK, serial, or RealSense imports.

The local test suite validates these assertions. Physical validation remains a
separately authorized future phase.
