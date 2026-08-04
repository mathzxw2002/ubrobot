"""Tests for the Operator Console camera view panel (M8 robot-eye view).

Covers the live-camera display state machine: frame available -> 正常,
camera channel connected without a frame -> 等待画面, edge unreachable ->
不可用, plus the telemetry metadata normalization that keeps the camera
row's resolution visible after the console's own refresh overwrites the
Robot Edge camera_info payload.
"""

from __future__ import annotations

import unittest

try:
    from chat_ui.app import (
        _camera_view_markdown,
        _channel_summary,
    )
    from chat_ui.pipeline import ChatPipeline
    HAS_VIEW = True
except ImportError:
    HAS_VIEW = False


def _camera_sample(*, available: bool = True, disconnected: bool = False) -> dict:
    state = "disconnected" if disconnected else ("available" if available else "unavailable")
    return {
        "channel": "camera",
        "state": state,
        "available": available,
        "disconnected": disconnected,
        "stale": False,
        "age_sec": 0.5,
        "value": {"width": 640, "height": 480},
    }


def _snapshot(camera: dict) -> dict:
    return {"telemetry": {"camera": camera}}


@unittest.skipUnless(HAS_VIEW, "chat_ui.app not importable")
class TestCameraViewMarkdown(unittest.TestCase):
    def test_live_frame_reports_resolution(self) -> None:
        from PIL import Image

        image = Image.new("RGB", (640, 480))
        text = _camera_view_markdown(image, _snapshot(_camera_sample()))
        self.assertIn("正常", text)
        self.assertIn("640×480", text)

    def test_waiting_when_connected_but_no_frame(self) -> None:
        text = _camera_view_markdown(None, _snapshot(_camera_sample()))
        self.assertIn("等待画面", text)

    def test_unavailable_when_edge_disconnected(self) -> None:
        text = _camera_view_markdown(
            None,
            _snapshot(_camera_sample(available=False, disconnected=True)),
        )
        self.assertIn("不可用", text)

    def test_unavailable_when_channel_missing(self) -> None:
        text = _camera_view_markdown(None, {"telemetry": {}})
        self.assertIn("不可用", text)


@unittest.skipUnless(HAS_VIEW, "chat_ui.app not importable")
class TestObservationMetadata(unittest.TestCase):
    def test_none_reports_unavailable(self) -> None:
        meta = ChatPipeline._observation_metadata(None)
        self.assertFalse(meta["available"])

    def test_pil_image_exposes_resolution(self) -> None:
        from PIL import Image

        meta = ChatPipeline._observation_metadata(Image.new("RGB", (640, 480)))
        self.assertTrue(meta["available"])
        self.assertEqual((meta["width"], meta["height"]), (640, 480))
        self.assertEqual(meta["size"], [640, 480])

    def test_missing_size_keeps_width_none(self) -> None:
        meta = ChatPipeline._observation_metadata(object())
        self.assertTrue(meta["available"])
        self.assertIsNone(meta["width"])
        self.assertIsNone(meta["height"])


@unittest.skipUnless(HAS_VIEW, "chat_ui.app not importable")
class TestChannelSummarySizeFallback(unittest.TestCase):
    def test_camera_summary_uses_width_height(self) -> None:
        self.assertEqual(
            _channel_summary("camera", {"width": 640, "height": 480}),
            "640×480",
        )

    def test_camera_summary_falls_back_to_size_list(self) -> None:
        self.assertEqual(
            _channel_summary("camera", {"size": [640, 480]}),
            "640×480",
        )

    def test_depth_summary_falls_back_to_size_list(self) -> None:
        self.assertEqual(
            _channel_summary("depth", {"size": [320, 240], "unit": "m"}),
            "320×240 · m",
        )


if __name__ == "__main__":
    unittest.main()
