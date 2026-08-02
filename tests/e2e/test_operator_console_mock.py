"""Process-level Operator Console acceptance tests without ROS or hardware."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from websockets.sync.client import connect


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "operator_scenarios.json"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class OperatorConsoleMockE2ETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenarios = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        cls.port = _free_port()
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        cls.shutdown_token = "m3-test-shutdown-token"
        cls.temp_dir = tempfile.TemporaryDirectory(prefix="ubrobot-m3-")
        cls.stdout_path = Path(cls.temp_dir.name) / "stdout.log"
        cls.stderr_path = Path(cls.temp_dir.name) / "stderr.log"
        cls.stdout_file = cls.stdout_path.open("w", encoding="utf-8")
        cls.stderr_file = cls.stderr_path.open("w", encoding="utf-8")
        env = os.environ.copy()
        env.update(
            {
                "UBROBOT_CHAT_BACKEND": "cortex-mock",
                "UBROBOT_CHAT_MEDIA": "off",
                "UBROBOT_VOICE_PROVIDER": "mock",
                "UBROBOT_CHAT_TLS": "off",
                "UBROBOT_CHAT_HOST": "127.0.0.1",
                "UBROBOT_CHAT_PORT": str(cls.port),
                "UBROBOT_CHAT_LOG_LEVEL": "WARNING",
                "UBROBOT_MOCK_NAV_DURATION_SEC": "0.4",
                "UBROBOT_MOCK_REPLY_DELAY_SEC": "0.02",
                "UBROBOT_SHUTDOWN_TOKEN": cls.shutdown_token,
                "PYTHONPATH": os.pathsep.join(
                    [str(ROOT / "src"), str(ROOT / "src" / "chat_ui")]
                ),
            }
        )
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        cls.process = subprocess.Popen(
            [sys.executable, "-u", str(ROOT / "src" / "chat_ui" / "app.py")],
            cwd=ROOT,
            env=env,
            stdout=cls.stdout_file,
            stderr=cls.stderr_file,
            creationflags=creationflags,
        )
        deadline = time.monotonic() + 30.0
        last_error = None
        while time.monotonic() < deadline:
            if cls.process.poll() is not None:
                break
            try:
                status, payload = cls._request("GET", "/api/health/ready")
                if status == 200 and payload.get("status") == "ready":
                    return
            except (URLError, ConnectionError) as exc:
                last_error = exc
            time.sleep(0.1)
        cls.stdout_file.flush()
        cls.stderr_file.flush()
        stderr = cls.stderr_path.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(
            f"Operator Console failed to start on {cls.port}: {last_error}\n{stderr[-4000:]}"
        )

    @classmethod
    def tearDownClass(cls):
        process = getattr(cls, "process", None)
        if process is not None and process.poll() is None:
            try:
                cls._request(
                    "POST",
                    "/api/admin/shutdown",
                    headers={"X-UBRobot-Shutdown-Token": cls.shutdown_token},
                )
                process.wait(timeout=15)
            except Exception:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        for name in ("stdout_file", "stderr_file"):
            file = getattr(cls, name, None)
            if file is not None:
                file.close()
        temp_dir = getattr(cls, "temp_dir", None)
        if temp_dir is not None:
            temp_dir.cleanup()

    @classmethod
    def _request(cls, method, path, payload=None, headers=None):
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request_headers = {"Content-Type": "application/json", **(headers or {})}
        request = Request(
            cls.base_url + path,
            data=body,
            headers=request_headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    @classmethod
    def _snapshot(cls):
        status, payload = cls._request("GET", "/api/operator/snapshot")
        if status != 200:
            raise AssertionError(payload)
        return payload

    @classmethod
    def _wait_for_active(cls, timeout=2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            active = cls._snapshot()["snapshot"]["tasks"]["active_task"]
            if active is not None:
                return active
            time.sleep(0.01)
        raise AssertionError("mock task did not become active")

    def test_01_full_navigation_succeeds_and_timeline_is_retained(self):
        scenario = self.scenarios["navigation_success"]
        status, result = self._request(
            "POST",
            "/api/operator/interactions",
            {"text": scenario["command"], "source": scenario["source"]},
        )

        self.assertEqual(status, 200)
        self.assertTrue(result["dispatched"])
        self.assertEqual(result["turn"]["category"], "new_task")
        snapshot = self._snapshot()["snapshot"]
        task = next(task for task in snapshot["tasks"]["tasks"] if task["task_id"] == result["task_id"])
        self.assertEqual(task["status"], "succeeded")
        task_events = [
            event
            for event in snapshot["tasks"]["events"]
            if event["task_id"] == result["task_id"]
        ]
        kinds = [event["event_type"] for event in task_events]
        for expected in scenario["expected_task_events"]:
            self.assertIn(expected, kinds)
        messages = "\n".join(event["message"] for event in task_events)
        for fragment in scenario["expected_feedback_fragments"]:
            self.assertIn(fragment, messages)

    def test_02_status_queue_cancel_and_emergency_paths(self):
        navigation = self.scenarios["navigation_success"]
        outcome = {}

        def submit_navigation():
            outcome["navigation"] = self._request(
                "POST",
                "/api/operator/interactions",
                {"text": navigation["command"], "source": "text"},
            )

        worker = threading.Thread(target=submit_navigation)
        worker.start()
        active = self._wait_for_active()

        status_scenario = self.scenarios["status_query"]
        status, query = self._request(
            "POST",
            "/api/operator/interactions",
            {"text": status_scenario["command"], "source": status_scenario["source"]},
        )
        self.assertEqual(status, 200)
        self.assertEqual(query["turn"]["category"], status_scenario["category"])
        self.assertFalse(query["dispatched"])
        self.assertEqual(query["task_id"], active["task_id"])

        queued_scenario = self.scenarios["queued_task"]
        status, queued = self._request(
            "POST",
            "/api/operator/interactions",
            {"text": queued_scenario["command"], "source": queued_scenario["source"]},
        )
        self.assertEqual(status, 200)
        self.assertFalse(queued["dispatched"])

        status, cancelled = self._request("POST", "/api/operator/cancel")
        self.assertEqual(status, 200)
        self.assertTrue(cancelled["acknowledged"])
        worker.join(2.0)
        self.assertFalse(worker.is_alive())
        self.assertEqual(outcome["navigation"][0], 409)

        status, stopped = self._request("POST", "/api/operator/emergency-stop")
        self.assertEqual(status, 200)
        self.assertFalse(stopped["voice"]["session_active"])
        snapshot = self._snapshot()["snapshot"]
        queued_task = next(
            task for task in snapshot["tasks"]["tasks"] if task["task_id"] == queued["task_id"]
        )
        self.assertEqual(queued_task["status"], "superseded")
        safety = [
            event
            for event in snapshot["tasks"]["events"]
            if event["event_type"] == "safety.emergency_stop"
        ]
        self.assertEqual(safety[-1]["data"]["priority"], "critical")

    def test_03_operator_and_voice_websockets_reconnect(self):
        cursor = self._snapshot()["latest_event_id"]
        operator_url = f"ws://127.0.0.1:{self.port}/api/operator/events?after={cursor}"
        with connect(operator_url, open_timeout=3) as websocket:
            initial = json.loads(websocket.recv(timeout=3))
            self.assertEqual(initial["type"], "snapshot")
            self._request(
                "POST",
                "/api/operator/interactions",
                {"text": "任务进度怎么样？", "source": "text"},
            )
            observed = []
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and "interaction.completed" not in observed:
                message = json.loads(websocket.recv(timeout=1))
                if message["type"] == "event":
                    observed.append(message["event"]["kind"])
            self.assertIn("interaction.received", observed)
            self.assertIn("interaction.completed", observed)

        voice_url = f"ws://127.0.0.1:{self.port}/api/voice/stream"
        for payload in (b"first-session", b"reconnected-session"):
            with connect(voice_url, open_timeout=3) as websocket:
                websocket.send(payload)
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                voice = self._snapshot()["snapshot"]["voice"]
                if voice["state"] == "idle" and not voice["session_active"]:
                    break
                time.sleep(0.01)
            self.assertEqual(voice["state"], "idle")


if __name__ == "__main__":
    unittest.main()
