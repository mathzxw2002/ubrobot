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
        active_joint_indices: tuple[int, ...] | None = None,
        max_iterations: int = 50,
        tolerance: float = 1e-3,
        damping: float = 1e-2,
        max_step: float = 0.2,
        position_tolerance: float | None = None,
        orientation_weight: float = 0.1,
    ) -> None:
        import pinocchio  # noqa: PLC0415 - deferred

        self._pinocchio = pinocchio
        self._model = pinocchio.buildModelFromUrdf(urdf_path)
        self._data = self._model.createData()
        # Only the six revolute arm joints are optimized. joint7 (tiny
        # revolute) and joint8 (gripper prismatic) stay locked at neutral so
        # the DLS never wastes steps on near-zero-range columns.
        self._active = list(active_joint_indices or range(6))
        self._max_iterations = int(max_iterations)
        self._tolerance = float(tolerance)
        self._damping = float(damping)
        self._max_step = float(max_step)
        self._position_tolerance = (
            float(position_tolerance) if position_tolerance is not None else float(tolerance)
        )
        self._orientation_weight = float(orientation_weight)

    def solve(self, position, orientation) -> list[float]:
        """Return six joint angles for ``position`` + quaternion.

        Uses damped least-squares inverse kinematics with the tool frame's
        WORLD-frame Jacobian (6 x nq), so position and orientation errors are
        expressed in the same frame as the Jacobian. The damped pseudo-inverse
        ``Jt @ inv(J Jt + lambda I)`` produces an nq-sized step. Rotation
        error is ``log3(R_target R_tool^T)``.
        """
        import numpy as np  # noqa: PLC0415 - deferred

        pino = self._pinocchio
        model = self._model
        data = self._data
        target = np.array([float(v) for v in position])
        quat = np.array([float(v) for v in (orientation or (0, 0, 0, 1))])

        q = pino.neutral(model)
        tool_id = len(model.frames) - 1
        for _ in range(self._max_iterations):
            pino.forwardKinematics(model, data, q)
            pino.updateFramePlacements(model, data)
            oMtool = data.oMf[tool_id]
            err_lin = target - oMtool.translation
            # Position is the primary grasp constraint; orientation is
            # secondary (a gripper tolerates modest approach-angle error).
            # A 6-DOF uniform error dominated by orientation diverges when a
            # joint hits its limit, so weight rotation lower.
            err_rot = self._orientation_weight * pino.log3(
                oMtool.rotation.T @ _quat_to_rot(quat)
            )
            error = np.concatenate([err_lin, err_rot])
            if np.linalg.norm(err_lin) < self._position_tolerance:
                return [float(v) for v in q]
            J = self._joint_jacobian(q)
            J_active = J[:, self._active]  # (6, n_active)
            rows, cols = J_active.shape
            if cols == 0 or rows == 0:
                raise RuntimeError("empty Jacobian for Piper tool frame")
            Jt = J_active.T
            damped = J_active @ Jt + self._damping * np.eye(rows)
            step_active = Jt @ np.linalg.inv(damped) @ error
            # Normalize the step so the DLS never over-shoots into a
            # non-convergent oscillation (robust to singular Jacobians).
            norm_step = np.linalg.norm(step_active)
            if norm_step > self._max_step:
                step_active = step_active * (self._max_step / norm_step)
            step = np.zeros(model.nq)
            step[self._active] = step_active
            q = np.clip(q + step, model.lowerPositionLimit, model.upperPositionLimit)
        raise RuntimeError(
            f"IK did not converge within {self._max_iterations} iterations for {position}"
        )

    def _joint_jacobian(self, q) -> Any:
        """Return the tool frame's WORLD-frame Jacobian (6 x nq).

        pinocchio 3.9+ exposes ``getFrameJacobian(model, data, frame_id,
        ReferenceFrame.WORLD)`` after ``computeJointJacobians``; this keeps
        the error and Jacobian in the same frame. Older versions fall back to
        ``computeJointJacobian(model, data, q, JOINT_MODEL, joint_id)``.
        """
        pino = self._pinocchio
        tool_id = len(self._model.frames) - 1
        joint_id = self._model.frames[tool_id].parentJoint
        if hasattr(pino, "getFrameJacobian"):
            pino.computeJointJacobians(self._model, self._data, q)
            return pino.getFrameJacobian(
                self._model, self._data, tool_id, pino.ReferenceFrame.WORLD
            )
        return pino.computeJointJacobian(
            self._model, self._data, q, pino.JOINT_MODEL, joint_id
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
