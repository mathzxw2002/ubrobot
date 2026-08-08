"""Pure motion-authority tracking (shared, no ROS imports).

``AuthorityTracker`` is the fail-closed evidence source shared by the ROS
grasp server (``ubrobot_manipulation.authority`` re-exports this class) and the
Robot Edge motion arbiter (``robot_edge.motion_arbitration``). It lives in
``ubrobot_contracts`` so the pure-Python workstation layer never depends on a
ROS ament package.

The grasp server must fail closed: without fresh positive evidence that
navigation holds no command lease and the base is publishing zero velocity,
grasping is not allowed. This tracker owns that evidence; the ROS adapter
only feeds it samples and a clock.
"""

import math
from collections import deque

LEASE_MAX_AGE_SEC = 0.5
CMD_VEL_WINDOW_SEC = 0.5
CMD_VEL_EPSILON = 1.0e-4


class AuthorityTracker:
    """Fail-closed view of navigation lease and base-velocity evidence."""

    def __init__(
        self,
        *,
        lease_max_age_sec: float = LEASE_MAX_AGE_SEC,
        cmd_vel_window_sec: float = CMD_VEL_WINDOW_SEC,
        cmd_vel_epsilon: float = CMD_VEL_EPSILON,
    ):
        if lease_max_age_sec <= 0 or cmd_vel_window_sec <= 0:
            raise ValueError("freshness windows must be positive")
        self._lease_max_age_sec = float(lease_max_age_sec)
        self._cmd_vel_window_sec = float(cmd_vel_window_sec)
        self._cmd_vel_epsilon = float(cmd_vel_epsilon)
        self._lease_active_since: float | None = None
        self._lease_cleared_at: float | None = None
        self._cmd_vel_samples: deque = deque()

    # ----------------------------------------------------------- producers

    def note_lease(self, lease_id: str, now: float) -> None:
        """Record one lease heartbeat; an empty string revokes authority."""
        if lease_id:
            self._lease_active_since = float(now)
        else:
            self._lease_active_since = None
            self._lease_cleared_at = float(now)

    def note_cmd_vel(self, x: float, y: float, z: float, now: float) -> None:
        try:
            magnitude = max(abs(float(x)), abs(float(y)), abs(float(z)))
        except (TypeError, ValueError):
            magnitude = math.inf
        if not math.isfinite(magnitude):
            magnitude = math.inf
        self._cmd_vel_samples.append((float(now), magnitude))
        self._prune(float(now))

    # ----------------------------------------------------------- consumers

    def navigation_lease_active(self, now: float) -> bool:
        """True only while fresh heartbeats keep the lease alive."""
        if self._lease_active_since is None:
            return False
        return (float(now) - self._lease_active_since) <= self._lease_max_age_sec

    def base_is_stationary(self, now: float) -> bool:
        """True only with recent, all-zero velocity evidence.

        No recent samples means no evidence of stillness, which fails
        closed (not stationary).
        """
        now = float(now)
        self._prune(now)
        if not self._cmd_vel_samples:
            return False
        return all(
            magnitude <= self._cmd_vel_epsilon
            for _ts, magnitude in self._cmd_vel_samples
        )

    # ------------------------------------------------------------- helpers

    def _prune(self, now: float) -> None:
        cutoff = now - self._cmd_vel_window_sec
        while self._cmd_vel_samples and self._cmd_vel_samples[0][0] < cutoff:
            self._cmd_vel_samples.popleft()
