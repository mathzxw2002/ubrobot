# ADR-0003: Use Cortex with a ROS 2 navigation capability layer

## Status

Accepted

## Context

UBRobot currently routes user requests in `Go2Manager.agent_response()` by
matching prefixes such as `nav:` and `grasp:`. The working EMOS deployment can
detect a vision target, run Kompass vision-depth control, and publish `/cmd_vel`
to the independent LeKiwi driver container. A previous real-motion test proved
that a client-side timeout can leave the downstream Kompass action goal alive;
the stale goal then resumes motion when the hardware driver starts.

The navigation integration must therefore support natural-language planning
without giving the LLM direct access to velocity commands, and it must make
goal ownership, cancellation, timeout, and command authority explicit.

## Decision

Use EMOS Cortex as the high-level orchestrator. Expose navigation through a
repository-owned `NavigateToObject` ROS 2 Action in a new
`ubrobot_navigation` package. The capability server delegates to Kompass'
`/track_vision_target` action and owns the complete downstream goal lifecycle.

Kompass publishes raw navigation commands on `/navigation/raw_cmd_vel`, not on
the hardware-facing `/cmd_vel`. A deterministic command guard forwards raw
commands only while it receives a valid, short-lived heartbeat for the active
outer action goal. Cancellation, timeout, capability-process failure, stale raw
commands, or loss of the heartbeat produces zero velocity. The LeKiwi driver
retains its independent limits and 250 ms watchdog.

The first release supports vision-target navigation only. Point navigation,
map navigation, manipulation, and direct hardware controls are out of scope.

## Consequences

### Positive

- Cortex plans with a semantic navigation tool rather than raw motor commands.
- Every navigation request has feedback, cancellation, timeout, and a result.
- A stale Kompass goal cannot command the driver without a live outer-goal
  lease.
- EMOS, the capability layer, and the LeKiwi driver remain independently
  testable and deployable.
- The same pattern can later be reused for Piper manipulation capabilities.

### Negative

- Adds two ROS nodes and one custom action definition.
- Adds another command path that must be observed and tested.
- EMOS 0.7.0 Cortex APIs must be compatibility-tested before the recipe is
  changed; current online documentation describes 0.7.6.

### Neutral

- The existing `/track_vision_target` action remains the implementation behind
  the new stable API.
- The existing `vision_depth_follower` recipe remains available as a rollback.

## Alternatives Considered

### Let Cortex call `/track_vision_target` directly

Rejected because it preserves the stale-goal failure mode and couples Cortex to
Kompass-specific goal and feedback schemas.

### Register existing `Go2Manager` methods as Python tools

Rejected because it keeps orchestration and hardware-adjacent state in one
process, provides weak cancellation semantics, and prolongs prefix-based code.

### Introduce a generic HTTP or MCP capability gateway

Deferred because ROS 2 Action already supplies feedback, cancellation, and
graph discovery on the robot. A network-neutral gateway can be added later if
capabilities must be controlled outside the ROS domain.

## References

- `docs/validation/2026-07-29-emos-real-motion-integration.md`
- `deploy/emos/recipes/vision_depth_follower/recipe.py`
- https://emos.automatikarobotics.com/recipes/planning-and-manipulation/cortex-agent.html

