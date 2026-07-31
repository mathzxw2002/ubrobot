"""Contract and fixture-behavior tests for the M1 end-to-end mock validation."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy" / "emos" / "test"))

from mock_planner_server import (  # noqa: E402
    FixtureConfig,
    NAVIGATION_TOOL_NAME,
    decide_response,
)


def chat_payload(messages, tools=None):
    payload = {"model": "mock-planner", "messages": messages}
    if tools is not None:
        payload["tools"] = tools
    return payload


class MockPlannerFixtureTest(unittest.TestCase):
    def setUp(self):
        self.config = FixtureConfig(target="chair", timeout_sec=20.0)

    def test_navigation_prompt_returns_one_navigation_tool_call(self):
        message = decide_response(
            chat_payload([{"role": "user", "content": "请走到椅子旁边"}]),
            self.config,
        )
        tool_calls = message.get("tool_calls")
        self.assertIsNotNone(tool_calls)
        self.assertEqual(len(tool_calls), 1)
        function = tool_calls[0]["function"]
        self.assertEqual(function["name"], NAVIGATION_TOOL_NAME)
        # EMOS Cortex `_parse_tool_args` requires the already-parsed object
        # form; the OpenAI JSON-string form crashes it upstream.
        self.assertEqual(
            function["arguments"], {"target": "chair", "timeout_sec": 20.0}
        )

    def test_english_navigation_prompt_also_triggers(self):
        message = decide_response(
            chat_payload([{"role": "user", "content": "navigate to the chair"}]),
            self.config,
        )
        self.assertEqual(
            message["tool_calls"][0]["function"]["name"], NAVIGATION_TOOL_NAME
        )

    def test_plain_prompt_returns_text_without_tools(self):
        # The fixture matches a simple pattern and intentionally does not
        # parse negation, so use a genuinely non-navigation prompt here.
        message = decide_response(
            chat_payload([{"role": "user", "content": "报告编排状态"}]),
            self.config,
        )
        self.assertNotIn("tool_calls", message)
        self.assertIn("报告编排状态", message["content"])

    def test_confirmation_with_active_action_returns_continue(self):
        confirmation = (
            "Original plan:\n"
            "  1. send_goal_to__ubrobot_navigation_navigate_to_object [NEXT]\n"
            "\nNext action: send_goal_to__ubrobot_navigation_navigate_to_object"
            " with arguments {'target': 'chair'}\n\n"
            "[Active Tools Status]\n"
            "- send_goal_to__ubrobot_navigation_navigate_to_object: active "
            "(running for 1.2s)\n[End Of Tools Status Update]\n\n"
            "Respond EXECUTE, SKIP, ABORT, or CONTINUE."
        )
        message = decide_response(
            chat_payload([{"role": "user", "content": confirmation}]),
            self.config,
        )
        self.assertEqual(message, {"role": "assistant", "content": "CONTINUE"})

    def test_confirmation_without_active_action_returns_execute(self):
        # The confirmation text mentions the navigation tool name; it must
        # not be mistaken for a navigation request.
        confirmation = (
            "Original plan:\n"
            "  1. send_goal_to__ubrobot_navigation_navigate_to_object [NEXT]\n"
            "\nNext action: send_goal_to__ubrobot_navigation_navigate_to_object"
            "\n\nRespond EXECUTE, SKIP, ABORT, or CONTINUE."
        )
        message = decide_response(
            chat_payload([{"role": "user", "content": confirmation}]),
            self.config,
        )
        self.assertEqual(message, {"role": "assistant", "content": "EXECUTE"})

    def test_executed_tool_result_yields_final_text(self):
        messages = [
            {"role": "user", "content": "请走到椅子旁边"},
            {"role": "assistant", "content": None, "tool_calls": []},
            {"role": "tool", "tool_call_id": "exec_0", "content": "SUCCEEDED"},
        ]
        message = decide_response(chat_payload(messages), self.config)
        self.assertNotIn("tool_calls", message)
        self.assertIn("chair", message["content"])

    def test_fixture_is_deterministic(self):
        payload = chat_payload([{"role": "user", "content": "请走到椅子旁边"}])
        self.assertEqual(
            decide_response(payload, self.config),
            decide_response(payload, self.config),
        )


class EndToEndDeploymentContractTest(unittest.TestCase):
    def test_image_contains_test_utilities_and_ui_transport(self):
        dockerfile = (ROOT / "deploy/emos/Dockerfile").read_text(encoding="utf-8")
        self.assertIn(
            "COPY src/chat_ui/cortex_client.py /opt/ubrobot/test/cortex_client.py",
            dockerfile,
        )
        self.assertIn("COPY deploy/emos/test/ /opt/ubrobot/test/", dockerfile)

    def test_end_to_end_harness_uses_production_transport_and_fixture(self):
        harness = (ROOT / "deploy/emos/test/end_to_end_mock_test.py").read_text(
            encoding="utf-8"
        )
        for token in (
            "create_ros_cortex_client",
            "DeterministicTrackingFixture",
            "assert_forward_signature",
            "STOP_DEADLINE_SEC",
            NAVIGATION_TOOL_NAME,
            "cancel_active",
        ):
            self.assertIn(token, harness)

    def test_run_script_keeps_mock_containers_device_free(self):
        script = (ROOT / "deploy/emos/test/run_end_to_end_mock.sh").read_text(
            encoding="utf-8"
        )
        for token in (
            "hardware_mode:=mock",
            "start_sensors:=false",
            "--cortex-only",
            "mock_planner_server.py",
            "end_to_end_mock_test.py",
            "/etc/fastdds/udp-only.xml",
        ):
            self.assertIn(token, script)
        # No container may map a hardware device in the mock validation.
        self.assertNotIn("--device", script)
        self.assertNotIn("/dev/lekiwi-base", script)
        self.assertNotIn("--privileged", script)


if __name__ == "__main__":
    unittest.main()
