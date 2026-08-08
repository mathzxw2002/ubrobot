"""Tests for centralized runtime settings (ubrobot_contracts.settings).

Verifies env-prefix mapping, defaults, validation, and that the settings
classes are pure Python (no ROS/hardware imports).
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from ubrobot_contracts.settings import ConsoleSettings, EdgeSettings


class ConsoleSettingsTest(unittest.TestCase):
    def test_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            s = ConsoleSettings()
        self.assertEqual(s.host, "0.0.0.0")
        self.assertEqual(s.port, 7863)
        self.assertEqual(s.backend, "cortex")
        self.assertTrue(s.media)
        self.assertTrue(s.tls)

    def test_env_prefix_maps_fields(self) -> None:
        with patch.dict(
            os.environ,
            {
                "UBROBOT_CHAT_HOST": "10.0.0.5",
                "UBROBOT_CHAT_PORT": "9000",
                "UBROBOT_CHAT_BACKEND": "robot-edge",
                "UBROBOT_CHAT_MEDIA": "off",
                "UBROBOT_CHAT_TLS": "off",
                "UBROBOT_EDGE_URL": "http://10.0.0.6:8780",
                "UBROBOT_VOICE_PROVIDER": "mock",
                "DASHSCOPE_API_KEY": "sk-test-123",
            },
            clear=True,
        ):
            s = ConsoleSettings()
        self.assertEqual(s.host, "10.0.0.5")
        self.assertEqual(s.port, 9000)
        self.assertEqual(s.backend, "robot-edge")
        self.assertFalse(s.media)
        self.assertFalse(s.tls)
        self.assertEqual(s.edge_url, "http://10.0.0.6:8780")
        self.assertEqual(s.voice_provider, "mock")
        self.assertEqual(s.dashscope_api_key, "sk-test-123")

    def test_backend_validation_rejects_unknown(self) -> None:
        with patch.dict(
            os.environ, {"UBROBOT_CHAT_BACKEND": "bogus"}, clear=True
        ):
            with self.assertRaises(ValueError):
                ConsoleSettings()

    def test_port_validation(self) -> None:
        with patch.dict(os.environ, {"UBROBOT_CHAT_PORT": "99999"}, clear=True):
            with self.assertRaises(ValueError):
                ConsoleSettings()


class EdgeSettingsTest(unittest.TestCase):
    def test_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            s = EdgeSettings()
        self.assertEqual(s.mode, "fixture")
        self.assertFalse(s.hardware_authority)
        self.assertFalse(s.estop_enabled)
        self.assertEqual(s.request_max_age_sec, 300)
        self.assertEqual(s.nonce_ttl_sec, 600)

    def test_env_prefix_maps_fields(self) -> None:
        with patch.dict(
            os.environ,
            {
                "UBROBOT_EDGE_MODE": "hardware",
                "UBROBOT_EDGE_HARDWARE_AUTHORITY": "true",
                "UBROBOT_EDGE_ESTOP_ENABLED": "true",
                "UBROBOT_EDGE_ESTOP_CHIP": "gpiochip0",
                "UBROBOT_EDGE_ESTOP_LINE": "4",
                "UBROBOT_EDGE_PORT": "9000",
                "UBROBOT_PLATFORM": "go2_piper",
            },
            clear=True,
        ):
            s = EdgeSettings()
        self.assertEqual(s.mode, "hardware")
        self.assertTrue(s.hardware_authority)
        self.assertTrue(s.estop_enabled)
        self.assertEqual(s.estop_chip, "gpiochip0")
        self.assertEqual(s.estop_line, "4")
        self.assertEqual(s.port, 9000)
        self.assertEqual(s.platform, "go2_piper")

    def test_mode_validation_rejects_unknown(self) -> None:
        with patch.dict(os.environ, {"UBROBOT_EDGE_MODE": "turbo"}, clear=True):
            with self.assertRaises(ValueError):
                EdgeSettings()

    def test_non_positive_timeouts_rejected(self) -> None:
        with patch.dict(
            os.environ, {"UBROBOT_EDGE_REQUEST_MAX_AGE_SEC": "0"}, clear=True
        ):
            with self.assertRaises(ValueError):
                EdgeSettings()


if __name__ == "__main__":
    unittest.main()
