"""Tests for the Operator Console telemetry Markdown renderer.

Covers both value shapes that reach the UI: the Robot Edge bridge envelope
(``{"channel", "state", "available", "source", "value": {...}}``) and the
flat fixture DTO dicts. The renderer must never leak secrets and must render
the actual sensor values (odometry, joints, camera metadata, lease,
capability health).
"""

from __future__ import annotations

import unittest

try:
    from chat_ui.app import (
        _channel_summary,
        _telemetry_markdown,
        _telemetry_value_fields,
    )
    HAS_RENDERER = True
except ImportError:
    HAS_RENDERER = False


def _sample(channel: str, *, value: dict, state: str = "available", age_sec: float | None = 0.3) -> dict:
    # stale still means data is present (aged); only disconnected/unavailable
    # reports available=False.
    available = state in ("available", "stale")
    return {
        "channel": channel,
        "value": value,
        "sequence": 1,
        "timestamp": "2026-08-03T00:00:00+00:00",
        "age_sec": age_sec,
        "state": state,
        "available": available,
        "stale": state == "stale",
        "disconnected": state == "disconnected",
    }


@unittest.skipUnless(HAS_RENDERER, "chat_ui.app not importable")
class TestTelemetryValueFields(unittest.TestCase):
    def test_unwraps_robot_edge_envelope(self) -> None:
        value = {
            "channel": "odometry",
            "state": "available",
            "available": True,
            "source": "robot-edge",
            "value": {"x": 1.25, "y": -0.5, "source": "fixture"},
        }
        fields = _telemetry_value_fields(value)
        self.assertEqual(fields.get("x"), 1.25)
        self.assertEqual(fields.get("y"), -0.5)

    def test_passes_flat_fixture_dto_through(self) -> None:
        value = {"channel": "camera", "state": "available", "width": 640, "height": 480}
        fields = _telemetry_value_fields(value)
        self.assertEqual(fields.get("width"), 640)

    def test_non_dict_value_returns_empty(self) -> None:
        self.assertEqual(_telemetry_value_fields(None), {})
        self.assertEqual(_telemetry_value_fields("nope"), {})
        self.assertEqual(_telemetry_value_fields([1, 2]), {})


@unittest.skipUnless(HAS_RENDERER, "chat_ui.app not importable")
class TestChannelSummary(unittest.TestCase):
    def test_camera_resolution(self) -> None:
        self.assertEqual(
            _channel_summary("camera", {"width": 640, "height": 480}),
            "640×480",
        )

    def test_depth_with_unit_and_calibration(self) -> None:
        summary = _channel_summary(
            "depth",
            {"width": 640, "height": 480, "unit": "m", "calibrated": True},
        )
        self.assertIn("640×480", summary)
        self.assertIn("m", summary)
        self.assertIn("已标定", summary)

    def test_odometry_numbers(self) -> None:
        summary = _channel_summary(
            "odometry", {"x": 1.25, "y": -0.5, "yaw": 0.1, "vx": 0.05}
        )
        self.assertEqual(summary, "x=1.25, y=-0.50, yaw=0.10, vx=0.05")

    def test_odometry_partial_fields(self) -> None:
        self.assertEqual(_channel_summary("odometry", {"x": 0.0}), "x=0.00")
        self.assertEqual(_channel_summary("odometry", {}), "-")

    def test_joint_states_with_positions(self) -> None:
        summary = _channel_summary(
            "joint_states",
            {
                "names": ["left", "right", "back"],
                "positions": [0.1, 0.2, 0.3],
                "motor_count": 3,
            },
        )
        self.assertEqual(summary, "3 电机 [left:0.10, right:0.20, back:0.30]")

    def test_joint_states_truncates_many_motors(self) -> None:
        summary = _channel_summary(
            "joint_states",
            {
                "names": [f"j{i}" for i in range(6)],
                "positions": [float(i) for i in range(6)],
            },
        )
        self.assertIn("6 电机 [", summary)
        self.assertIn("…", summary)

    def test_joint_states_empty(self) -> None:
        self.assertEqual(_channel_summary("joint_states", {"names": [], "positions": []}), "-")

    def test_lease_no_owner(self) -> None:
        self.assertEqual(_channel_summary("navigation_lease", {"owner": None}), "无")

    def test_lease_with_owner(self) -> None:
        summary = _channel_summary(
            "navigation_lease",
            {"owner": "operator-live", "lease_id": "l-1", "expires_at": "2026-08-03T00:01:00+00:00"},
        )
        self.assertEqual(summary, "持有者=operator-live 至 2026-08-03T00:01:00")

    def test_capability_health_counts(self) -> None:
        summary = _channel_summary(
            "capability_health",
            {
                "capabilities": {
                    "navigation": {"health": "healthy"},
                    "grasp": {"health": "unhealthy"},
                }
            },
        )
        self.assertEqual(summary, "navigation:healthy · grasp:unhealthy")


@unittest.skipUnless(HAS_RENDERER, "chat_ui.app not importable")
class TestTelemetryMarkdown(unittest.TestCase):
    def _render(self, telemetry: dict) -> str:
        return _telemetry_markdown({"telemetry": telemetry})

    def test_overview_table_has_key_value_column(self) -> None:
        text = self._render(
            {"odometry": _sample("odometry", value={"x": 1.25, "y": 0.0, "source": "fixture"})}
        )
        self.assertIn("| 通道 | 状态 | 更新 | 关键值 |", text)
        self.assertIn("x=1.25, y=0.00", text)
        self.assertIn("正常", text)
        self.assertIn("0.3s 前", text)

    def test_detail_block_renders_values(self) -> None:
        text = self._render(
            {
                "joint_states": _sample(
                    "joint_states",
                    value={
                        "names": ["left", "right"],
                        "positions": [0.1, -0.2],
                        "velocities": [0.0, 0.0],
                        "motor_count": 2,
                        "source": "robot-edge:ros",
                    },
                )
            }
        )
        self.assertIn("⚙️ 关节状态", text)
        self.assertIn("- 关节: left, right", text)
        self.assertIn("- 位置: 0.10, -0.20", text)
        self.assertIn("- 电机数: 2", text)

    def test_disconnected_channel_shows_detail(self) -> None:
        text = self._render(
            {
                "camera": _sample(
                    "camera",
                    state="disconnected",
                    value={"state": "disconnected", "detail": "robot edge unreachable"},
                )
            }
        )
        self.assertIn("断连", text)
        self.assertIn("robot edge unreachable", text)

    def test_stale_channel_label(self) -> None:
        text = self._render(
            {"odometry": _sample("odometry", state="stale", value={"x": 1.0}, age_sec=9.0)}
        )
        self.assertIn("陈旧", text)
        self.assertIn("9.0s 前", text)

    def test_camera_metadata_in_detail(self) -> None:
        text = self._render(
            {
                "camera": _sample(
                    "camera",
                    value={
                        "width": 640,
                        "height": 480,
                        "encoding": "rgb8",
                        "frame_id": "camera_color_optical_frame",
                        "calibrated": True,
                        "source": "robot-edge:ros",
                    },
                )
            }
        )
        self.assertIn("- 宽度: 640", text)
        self.assertIn("- 高度: 480", text)
        self.assertIn("- 编码: rgb8", text)
        self.assertIn("- 已标定: 是", text)

    def test_all_six_channels_render(self) -> None:
        telemetry = {
            "camera": _sample("camera", value={"width": 640, "height": 480}),
            "depth": _sample("depth", value={"width": 640, "height": 480, "unit": "m"}),
            "odometry": _sample("odometry", value={"x": 0.0, "y": 0.0, "yaw": 0.0}),
            "joint_states": _sample("joint_states", value={"names": [], "positions": []}),
            "navigation_lease": _sample("navigation_lease", value={"owner": None}),
            "capability_health": _sample("capability_health", value={"capabilities": {}}),
        }
        text = self._render(telemetry)
        for title in ("相机 RGB", "深度相机", "里程计", "关节状态", "导航租约", "能力健康"):
            self.assertIn(title, text)

    def test_no_secrets_or_sdk_objects_in_output(self) -> None:
        telemetry = {
            "navigation_lease": _sample(
                "navigation_lease",
                value={"owner": "operator", "lease_id": "lease-secret-12345"},
            ),
            "odometry": _sample("odometry", value={"x": 0.0}),
        }
        text = self._render(telemetry)
        self.assertNotIn("token", text)
        self.assertNotIn("Bearer", text)
        self.assertNotIn("<object", text)


if __name__ == "__main__":
    unittest.main()
