"""Tests for Robot Edge Prometheus metrics (P2)."""

from __future__ import annotations

import unittest

from robot_edge.metrics import METRICS, EdgeMetrics


class EdgeMetricsTest(unittest.TestCase):
    def test_registry_degrades_gracefully_without_client(self) -> None:
        # EdgeMetrics must import and render empty when prometheus_client is
        # absent (fixture/dev mode). The module-level singleton already exists.
        self.assertIsInstance(METRICS, EdgeMetrics)

    def test_render_returns_bytes(self) -> None:
        payload = METRICS.render()
        self.assertIsInstance(payload, bytes)
        # Either enabled (real payload) or disabled (empty), never None.
        if METRICS.enabled:
            self.assertGreater(len(payload), 0)
        else:
            self.assertEqual(payload, b"")

    def test_record_command_safe_when_disabled(self) -> None:
        # Must never raise even without prometheus_client.
        METRICS.record_command("succeeded")
        METRICS.set_lease_active(True)
        METRICS.set_safety_latched(False)
        METRICS.set_capability("navigation", True)


if __name__ == "__main__":
    unittest.main()
