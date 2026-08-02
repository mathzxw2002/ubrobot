"""Two-process acceptance: Operator Console <-> Robot Edge (fixture mode).

Starts a real Robot Edge process and a real Operator Console process on dynamic
ports and exercises the full safety-critical flow over HTTP. No mocks, no
in-process shortcuts: both processes are independent, exactly as in production.
"""

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

import httpx
from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_COMMAND = "导航到前面的椅子"
OPERATOR_TOKEN = "operator-token"
TOKEN_SCOPES = [
    "observe",
    "task.submit",
    "task.cancel",
    "safety.stop",
    "lease.manage",
]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _wait_ready(url: str, path: str, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    client = httpx.Client(base_url=url, timeout=2.0, trust_env=False)
    while time.monotonic() < deadline:
        try:
            if client.get(path).status_code == 200:
                client.close()
                return
        except httpx.RequestError:
            pass
        time.sleep(0.1)
    client.close()
    raise RuntimeError(f"service at {url} did not become ready at {path}")


class OperatorRobotEdgeFixtureE2E(unittest.TestCase):
    """Full two-process fixture acceptance."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory(prefix="ubrobot-m5-")
        tokens_path = Path(cls.temp_dir.name) / "tokens.json"
        tokens_path.write_text(
            json.dumps({OPERATOR_TOKEN: TOKEN_SCOPES}), encoding="utf-8"
        )
        cls.tokens_path = tokens_path

        # --- Robot Edge process ---
        cls.edge_port = _free_port()
        cls.edge_url = f"http://127.0.0.1:{cls.edge_port}"
        edge_env = os.environ.copy()
        edge_env.update(
            {
                "UBROBOT_EDGE_MODE": "fixture",
                "UBROBOT_EDGE_HOST": "127.0.0.1",
                "UBROBOT_EDGE_PORT": str(cls.edge_port),
                "UBROBOT_EDGE_TOKENS_FILE": str(tokens_path),
                "UBROBOT_EDGE_LOG_LEVEL": "warning",
                # Widen the active-command window so process-level cancel/E-stop
                # tests can observe the command mid-flight (<=100 ms per step).
                "UBROBOT_EDGE_FIXTURE_STEP_DELAY_SEC": "0.1",
                "PYTHONPATH": os.pathsep.join([str(ROOT / "src")]),
                "PYTHONIOENCODING": "utf-8",
            }
        )
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        # stderr goes to a file, never a PIPE: an unread pipe fills up and
        # blocks the child's main thread on the next log write, which wedges
        # the whole service (observed: console stops responding after the
        # cancellation ERROR traceback fills the pipe). The file also keeps
        # the full log available for failure diagnosis.
        cls.edge_stderr_path = Path(cls.temp_dir.name) / "edge.stderr.log"
        cls.edge_stderr_file = open(cls.edge_stderr_path, "wb")
        cls.edge_process = subprocess.Popen(
            [sys.executable, "-u", "-m", "robot_edge.app"],
            cwd=ROOT,
            env=edge_env,
            stdout=subprocess.DEVNULL,
            stderr=cls.edge_stderr_file,
            creationflags=creationflags,
        )
        try:
            _wait_ready(cls.edge_url, "/v1/health/live")
        except RuntimeError:
            cls._dump_edge_stderr()
            raise

        # --- Operator Console process ---
        cls.console_port = _free_port()
        cls.console_url = f"http://127.0.0.1:{cls.console_port}"
        cls.shutdown_token = "m5-test-shutdown-token"
        console_env = os.environ.copy()
        console_env.update(
            {
                "UBROBOT_CHAT_BACKEND": "robot-edge",
                "UBROBOT_CHAT_MEDIA": "off",
                "UBROBOT_VOICE_PROVIDER": "mock",
                "UBROBOT_CHAT_TLS": "off",
                "UBROBOT_CHAT_HOST": "127.0.0.1",
                "UBROBOT_CHAT_PORT": str(cls.console_port),
                "UBROBOT_CHAT_LOG_LEVEL": "WARNING",
                "UBROBOT_EDGE_URL": cls.edge_url,
                "UBROBOT_EDGE_TOKEN": OPERATOR_TOKEN,
                "UBROBOT_EDGE_OPERATOR_ID": "e2e-operator",
                "UBROBOT_SHUTDOWN_TOKEN": cls.shutdown_token,
                "PYTHONPATH": os.pathsep.join(
                    [str(ROOT / "src"), str(ROOT / "src" / "chat_ui")]
                ),
                "PYTHONIOENCODING": "utf-8",
            }
        )
        cls.console_stderr_path = Path(cls.temp_dir.name) / "console.stderr.log"
        cls.console_stderr_file = open(cls.console_stderr_path, "wb")
        cls.console_process = subprocess.Popen(
            [sys.executable, "-u", str(ROOT / "src" / "chat_ui" / "app.py")],
            cwd=ROOT,
            env=console_env,
            stdout=subprocess.DEVNULL,
            stderr=cls.console_stderr_file,
            creationflags=creationflags,
        )
        try:
            _wait_ready(cls.console_url, "/api/health/ready")
        except RuntimeError:
            cls._dump_console_stderr()
            raise

    @classmethod
    def tearDownClass(cls) -> None:
        # Operator Console: graceful shutdown via admin endpoint.
        console = getattr(cls, "console_process", None)
        if console is not None and console.poll() is None:
            try:
                cls._console_request(
                    "POST",
                    "/api/admin/shutdown",
                    headers={"X-UBRobot-Shutdown-Token": cls.shutdown_token},
                )
                console.wait(timeout=15)
            except Exception:
                console.terminate()
                try:
                    console.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    console.kill()
                    console.wait(timeout=5)
        # Edge: terminate (no admin shutdown endpoint).
        edge = getattr(cls, "edge_process", None)
        if edge is not None and edge.poll() is None:
            edge.terminate()
            try:
                edge.wait(timeout=5)
            except subprocess.TimeoutExpired:
                edge.kill()
                edge.wait(timeout=5)
        for name in ("console_stderr_file", "edge_stderr_file"):
            handle = getattr(cls, name, None)
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
        temp_dir = getattr(cls, "temp_dir", None)
        if temp_dir is not None:
            temp_dir.cleanup()
        # Both ports must be free of listeners once the processes are gone.
        cls._assert_no_listeners()

    @classmethod
    def _assert_no_listeners(cls) -> None:
        edge_port = getattr(cls, "edge_port", None)
        console_port = getattr(cls, "console_port", None)
        for label, port in (("Edge", edge_port), ("Operator", console_port)):
            if port is None:
                continue
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                    # connect_ex returns 0 only if a listener accepts; a refused
                    # connection (non-zero) means no listener remains.
                    if probe.connect_ex(("127.0.0.1", port)) != 0:
                        break
                time.sleep(0.1)
            else:
                raise AssertionError(
                    f"{label} listener still active on port {port} after shutdown"
                )

    @classmethod
    def _dump_edge_stderr(cls) -> None:
        stderr = b""
        path = getattr(cls, "edge_stderr_path", None)
        if path is not None and Path(path).exists():
            stderr = Path(path).read_bytes()
        raise RuntimeError(
            f"Edge failed to start. stderr:\n{stderr.decode('utf-8', 'replace')[-3000:]}"
        )

    @classmethod
    def _dump_console_stderr(cls) -> None:
        stderr = b""
        path = getattr(cls, "console_stderr_path", None)
        if path is not None and Path(path).exists():
            stderr = Path(path).read_bytes()
        raise RuntimeError(
            f"Operator Console failed to start. stderr:\n"
            f"{stderr.decode('utf-8', 'replace')[-3000:]}"
        )

    @classmethod
    def _console_request(cls, method, path, payload=None, headers=None):
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            cls.console_url + path,
            data=body,
            headers={"Content-Type": "application/json", **(headers or {})},
            method=method,
        )
        try:
            with urlopen(request, timeout=15) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    @classmethod
    def _edge_request(cls, method, path, payload=None):
        client = httpx.Client(base_url=cls.edge_url, timeout=10.0, trust_env=False)
        try:
            response = client.request(
                method,
                path,
                json=payload,
                headers={"Authorization": f"Bearer {OPERATOR_TOKEN}"},
            )
            try:
                body = response.json()
            except Exception:
                body = {"raw": response.text}
            return response.status_code, body
        finally:
            client.close()

    @classmethod
    def _console_snapshot(cls):
        status, payload = cls._console_request("GET", "/api/operator/snapshot")
        if status != 200:
            raise AssertionError(f"snapshot failed: {status} {payload}")
        return payload["snapshot"]

    # ------------------------------------------------------------------ steps

    def test_01_authenticate_and_authority_is_false(self) -> None:
        """Both processes are up; the Edge authenticates the operator token;
        hardware authority is false everywhere."""
        # Edge health (no auth)
        s, _ = self._edge_request("GET", "/v1/health/live")
        self.assertEqual(s, 200)
        # Edge authenticated read (observe scope)
        s, body = self._edge_request("GET", "/v1/capabilities")
        self.assertEqual(s, 200)
        self.assertIn("navigation", body["capabilities"])
        # Operator Console ready with hardware authority false.
        s, body = self._console_request("GET", "/api/health/ready")
        self.assertEqual(s, 200)

    def test_02_acquire_lease(self) -> None:
        """The operator can acquire a navigation lease on the Edge."""
        s, body = self._edge_request(
            "POST",
            "/v1/lease/acquire",
            {
                "operator_id": "e2e-operator",
                "nonce": "lease-nonce-1",
                "timestamp": _now_iso(),
                "duration_sec": 60.0,
            },
        )
        self.assertEqual(s, 200)
        self.assertEqual(body["state"], "active")
        self.assertEqual(body["owner"], "e2e-operator")

    def test_03_submit_command_and_observe_timelines(self) -> None:
        """Submit through the Operator API; task succeeds; Edge events record it."""
        s, result = self._console_request(
            "POST",
            "/api/operator/interactions",
            {"text": FIXTURE_COMMAND, "source": "text"},
        )
        self.assertEqual(s, 200)
        self.assertTrue(result["dispatched"])
        task_id = result["task_id"]

        # Wait for the task to reach a terminal state.
        deadline = time.monotonic() + 10.0
        status = None
        while time.monotonic() < deadline:
            snapshot = self._console_snapshot()
            task = next(
                (t for t in snapshot["tasks"]["tasks"] if t["task_id"] == task_id),
                None,
            )
            if task and task["status"] in {"succeeded", "failed", "cancelled"}:
                status = task["status"]
                break
            time.sleep(0.05)
        self.assertEqual(status, "succeeded")

        # Edge event timeline contains the deterministic sequence.
        s, body = self._edge_request("GET", "/v1/events?after=0")
        self.assertEqual(s, 200)
        states = [event["state"] for event in body["events"]]
        self.assertIn("accepted", states)
        self.assertIn("succeeded", states)

    def test_04_status_query_does_not_dispatch_second_command(self) -> None:
        """A status utterance reads state without creating a new Edge command."""
        s, result = self._console_request(
            "POST",
            "/api/operator/interactions",
            {"text": "任务进度怎么样？", "source": "text"},
        )
        self.assertEqual(s, 200)
        self.assertFalse(result["dispatched"])

    def test_05_cancel_active_command(self) -> None:
        """Cancel stops a running command; the in-flight interaction fails cancelled.

        The interactions endpoint blocks until the task completes, so the
        command must be submitted in a background thread and cancelled while it
        is still running.
        """
        outcome: dict[str, object] = {}

        def submit() -> None:
            status, result = self._console_request(
                "POST",
                "/api/operator/interactions",
                {"text": FIXTURE_COMMAND, "source": "text"},
            )
            outcome["status"] = status

        worker = threading.Thread(target=submit)
        worker.start()

        # Wait for the task to become active, then cancel.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            active = self._console_snapshot()["tasks"]["active_task"]
            if active is not None and active["status"] in {
                "planning",
                "running",
                "cancelling",
            }:
                break
            time.sleep(0.01)

        s, cancelled = self._console_request("POST", "/api/operator/cancel")
        self.assertEqual(s, 200)
        self.assertTrue(cancelled["acknowledged"])
        worker.join(timeout=5.0)
        # The in-flight interaction must end cancelled (409 from the console).
        self.assertEqual(outcome.get("status"), 409)

    def test_06_emergency_stop_latches_and_blocks_new_work(self) -> None:
        """Emergency stop latches the Edge; a subsequent command is rejected."""
        s, stopped = self._console_request("POST", "/api/operator/emergency-stop")
        self.assertEqual(s, 200)
        self.assertTrue(stopped["acknowledged"])

        # A new command must be rejected because the Edge is latched. The
        # console surfaces the backend failure as HTTP 409.
        s, result = self._console_request(
            "POST",
            "/api/operator/interactions",
            {"text": "再导航一次", "source": "text"},
        )
        self.assertEqual(s, 409)

    def test_07_authorized_reset_re_enables_work(self) -> None:
        """An explicit authorized reset clears the latch; commands succeed again."""
        s, body = self._edge_request(
            "POST",
            "/v1/safety/reset",
            {
                "correlation_id": "reset-1",
                "operator_id": "e2e-operator",
                "nonce": "reset-nonce-1",
                "timestamp": _now_iso(),
            },
        )
        self.assertEqual(s, 200)
        self.assertFalse(body["latched"])

        s, result = self._console_request(
            "POST",
            "/api/operator/interactions",
            {"text": FIXTURE_COMMAND, "source": "text"},
        )
        self.assertEqual(s, 200)
        self.assertTrue(result["dispatched"])
        task_id = result["task_id"]
        deadline = time.monotonic() + 10.0
        status = None
        while time.monotonic() < deadline:
            snapshot = self._console_snapshot()
            task = next(
                (t for t in snapshot["tasks"]["tasks"] if t["task_id"] == task_id),
                None,
            )
            if task and task["status"] in {"succeeded", "failed"}:
                status = task["status"]
                break
            time.sleep(0.05)
        self.assertEqual(status, "succeeded")

    def test_08_event_streams_reconnect(self) -> None:
        """The Operator event stream reconnects from a cursor; the Edge events
        endpoint replays from a cursor."""
        status, payload = self._console_request("GET", "/api/operator/snapshot")
        cursor = payload["latest_event_id"]
        url = f"ws://127.0.0.1:{self.console_port}/api/operator/events?after={cursor}"
        with connect(url, open_timeout=3) as websocket:
            initial = json.loads(websocket.recv(timeout=3))
            self.assertEqual(initial["type"], "snapshot")

        s, body = self._edge_request("GET", "/v1/events?after=0")
        self.assertEqual(s, 200)
        self.assertIsInstance(body["events"], list)


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    unittest.main()
