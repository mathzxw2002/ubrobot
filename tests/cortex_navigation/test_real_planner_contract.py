"""Contract tests for the M3 real-planner mock validation artifacts."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy" / "emos"))

import planner_relay  # noqa: E402


class CortexStringArgumentsPatchTest(unittest.TestCase):
    def test_dockerfile_patches_parse_tool_args(self):
        dockerfile = (ROOT / "deploy/emos/Dockerfile").read_text(encoding="utf-8")
        for token in (
            "cortex _parse_tool_args patch target not found",
            "isinstance(fn_args, str)",
            "json.loads(fn_args)",
        ):
            self.assertIn(token, dockerfile)

    def test_image_contains_planner_relay(self):
        dockerfile = (ROOT / "deploy/emos/Dockerfile").read_text(encoding="utf-8")
        self.assertIn(
            "COPY deploy/emos/planner_relay.py /opt/ubrobot/planner_relay.py",
            dockerfile,
        )


class PlannerRelayContractTest(unittest.TestCase):
    def test_relay_requires_https_upstream(self):
        with self.assertRaises(SystemExit):
            import os

            os.environ.pop("PLANNER_UPSTREAM_URL", None)
            planner_relay.main()

    def test_relay_has_no_hardcoded_credentials(self):
        source = (ROOT / "deploy/emos/planner_relay.py").read_text(encoding="utf-8")
        self.assertNotIn("api_key=", source.replace("PLANNER_API_KEY", ""))
        for forbidden in ("sk-", "Bearer ", "password"):
            self.assertNotIn(forbidden, source)
        self.assertIn("Authorization", source)  # header is forwarded, not set

    def test_relay_never_logs_bodies(self):
        source = (ROOT / "deploy/emos/planner_relay.py").read_text(encoding="utf-8")
        self.assertIn("Never log bodies", source)

    def test_relay_disables_upstream_compression(self):
        # Response bytes are forwarded verbatim without Content-Encoding, so
        # the upstream must never gzip (httpx would fail to decode).
        source = (ROOT / "deploy/emos/planner_relay.py").read_text(encoding="utf-8")
        self.assertIn('"accept-encoding",', source)
        self.assertIn('headers["Accept-Encoding"] = "identity"', source)

    def test_relay_maps_v1_prefix_to_upstream_base(self):
        handler_cls = planner_relay.make_handler(
            "https://ark.cn-beijing.volces.com/api/v3", 60.0
        )
        handler = handler_cls.__new__(handler_cls)
        handler.path = "/v1/chat/completions"
        self.assertEqual(
            handler._target_url(),
            "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
        )
        handler.path = "/v1/models"
        self.assertEqual(
            handler._target_url(),
            "https://ark.cn-beijing.volces.com/api/v3/models",
        )

    def test_relay_maps_v1_prefix_for_openai_style_base(self):
        handler_cls = planner_relay.make_handler("https://api.deepseek.com/v1", 60.0)
        handler = handler_cls.__new__(handler_cls)
        handler.path = "/v1/chat/completions"
        self.assertEqual(
            handler._target_url(),
            "https://api.deepseek.com/v1/chat/completions",
        )


class RealPlannerDeploymentTest(unittest.TestCase):
    def test_run_script_requires_credentials_from_environment(self):
        script = (ROOT / "deploy/emos/test/run_real_planner_mock.sh").read_text(
            encoding="utf-8"
        )
        for token in (
            "PLANNER_UPSTREAM_URL",
            "PLANNER_API_KEY",
            "PLANNER_CHECKPOINT",
            "REDACTED",
            "planner_relay.py",
            "real_planner_mock_test.py",
            "hardware_mode:=mock",
        ):
            self.assertIn(token, script)
        self.assertNotIn("--device", script)
        self.assertNotIn("/dev/lekiwi-base", script)
        self.assertNotIn("--privileged", script)

    def test_real_planner_client_is_behavioral(self):
        client = (ROOT / "deploy/emos/test/real_planner_mock_test.py").read_text(
            encoding="utf-8"
        )
        for token in (
            "create_ros_cortex_client",
            "run_non_motion",
            "run_cancel",
            "assert_forward_signature",
            "no feedback echoed the prompt",
        ):
            self.assertIn(token, client)


if __name__ == "__main__":
    unittest.main()
