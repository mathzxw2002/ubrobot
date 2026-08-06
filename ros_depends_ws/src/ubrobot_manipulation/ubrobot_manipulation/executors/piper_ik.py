"""Piper arm inverse kinematics (pinocchio, deferred import).

The IK solver maps a 6D grasp pose (position + orientation quaternion) to
six joint angles for ``piper_description.urdf``. pinocchio is imported
lazily so workstation tests never pull it in; the real solver is only
constructed inside ``PiperMotionBinding`` on the dock.
"""

from __future__ import annotations

from typing import Any


class PiperIk:
    """pinocchio-based IK for the Piper arm.

    Uses a damped least-squares (Levenberg-Marquardt style) closed-loop
    inverse kinematics. For the initial stationary-base grasp the seed is
    the current configuration; iteration stops on convergence or on a joint
    limit / iteration cap.
    """

    def __init__(
        self,
        *,
        urdf_path: str = "assets/urdf/piper_description.urdf",
        max_iterations: int = 50,
        tolerance: float = 1e-3,
        damping: float = 1e-2,
    ) -> None:
        import pinocchio  # noqa: PLC0415 - deferred

        self._pinocchio = pinocchio
        self._model = pinocchio.buildModelFromUrdf(urdf_path)
        self._data = self._model.createData()
        self._max_iterations = int(max_iterations)
        self._tolerance = float(tolerance)
        self._damping = float(damping)

    def solve(self, position, orientation) -> list[float]:
        """Return six joint angles for ``position`` + quaternion."""
        import numpy as np  # noqa: PLC0415 - deferred

        pino = self._pinocchio
        model = self._model
        data = self._data
        target = np.array([float(v) for v in position])
        quat = np.array([float(v) for v in (orientation or (0, 0, 0, 1))])

        # Seed from the current (home) configuration.
        q = pino.neutral(model)
        q = pino.randomConfiguration(model)
        for _ in range(self._max_iterations):
            pino.forwardKinematics(model, data, q)
            pino.updateFramePlacements(model, data)
            oMtool = data.oMf[len(model.frames) - 1]
            err_lin = target - oMtool.translation
            err_rot = pino.log3(oMtool.rotation.T @ _quat_to_rot(quat))
            error = np.concatenate([err_lin, err_rot])
            if np.linalg.norm(error) < self._tolerance:
                return [float(v) for v in q]
            J = pino.computeJointJacobian(model, data, q, pino.JOINT_MODEL)
            Jt = J.T
            step = Jt @ np.linalg.inv(J @ Jt + self._damping * np.eye(6)) @ error
            q = np.clip(q + step, model.lowerPositionLimit, model.upperPositionLimit)
        raise RuntimeError(
            f"IK did not converge within {self._max_iterations} iterations for {position}"
        )


def _quat_to_rot(quat: Any) -> Any:
    """Build a 3x3 rotation matrix from a quaternion (w,x,y,z)."""
    import numpy as np  # noqa: PLC0415 - deferred

    w, x, y, z = (float(v) for v in quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
