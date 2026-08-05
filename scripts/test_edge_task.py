#!/usr/bin/env python3
"""End-to-end navigation test: submit a task to Robot Edge and stream the timeline.

Mirrors scripts/test_edge_task.ps1 but uses Python so Chinese task text survives
the Windows argv -> HTTP -> ROS hop without GBK mangling.

Usage:
    python scripts/test_edge_task.py --task "导航到前面的椅子，移动10厘米以内"
    python scripts/test_edge_task.py --health-only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Windows console defaults to GBK; force UTF-8 so emoji/Chinese don't crash.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import urllib.request
import urllib.error

EDGE_HOST = "192.168.18.233"
EDGE_PORT = 8780
CONSOLE_PORT = 7863
TOKEN_FILE = Path(__file__).resolve().parent.parent / "tmp" / "edge_tokens.json"
TERMINAL = {"succeeded", "failed", "cancelled"}


def http(method: str, url: str, token: str, body: dict | None = None, timeout: float = 15.0):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=utf-8"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def load_token(submit: bool):
    tokens = json.loads(TOKEN_FILE.read_text(encoding="utf-8"))
    for tok, scopes in tokens.items():
        if submit and "task.submit" in scopes and "observe" in scopes:
            return tok
    for tok, scopes in tokens.items():
        if "observe" in scopes:
            return tok
    raise SystemExit("no suitable token found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="导航到前面的椅子，移动10厘米以内")
    ap.add_argument("--host", default=EDGE_HOST)
    ap.add_argument("--port", type=int, default=EDGE_PORT)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--health-only", action="store_true")
    args = ap.parse_args()

    edge = f"http://{args.host}:{args.port}"
    token = load_token(submit=not args.health_only)

    # 1. console health
    print("\n== Operator Console ==")
    try:
        h = http("GET", f"http://127.0.0.1:{CONSOLE_PORT}/api/health/ready", token, timeout=3)
        print(f"OK   status={h.get('status')} backend={h.get('backend')} mode={h.get('execution_mode')}")
    except Exception as e:
        print(f"WARN console: {e}")

    # 2. edge health + capabilities + camera
    print("\n== Robot Edge ==")
    try:
        r = http("GET", f"{edge}/v1/health/ready", token)
        print(f"OK   ready: {json.dumps(r, ensure_ascii=False)[:120]}")
    except Exception as e:
        print(f"FAIL edge health: {e}")
    try:
        c = http("GET", f"{edge}/v1/capabilities", token)
        print("OK   capabilities:", ", ".join(c.get("capabilities", {}).keys()))
    except Exception:
        pass

    if args.health_only:
        print("\nHealth checks done.")
        return

    # 3. submit task
    print(f"\n== Submit task: {args.task} ==")
    corr = str(uuid.uuid4())
    body = {
        "text": args.task,
        "correlation_id": corr,
        "operator_id": "manual-test-script",
        "nonce": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        acc = http("POST", f"{edge}/v1/commands", token, body, timeout=15)
    except urllib.error.HTTPError as e:
        print(f"FAIL submit HTTP {e.code}: {e.read().decode('utf-8','replace')[:200]}")
        sys.exit(1)
    cmd_id = acc["command_id"]
    print(f"accepted: command_id={cmd_id}")

    # 4. poll events
    print("\n== Event timeline ==")
    after = 0
    t0 = time.monotonic()
    terminal = None
    while time.monotonic() - t0 < args.timeout:
        try:
            ev = http("GET", f"{edge}/v1/events?after={after}", token, timeout=10)
        except Exception:
            time.sleep(0.5)
            continue
        for e in ev.get("events", []):
            after = max(after, int(e.get("sequence", after + 1)))
            if e.get("command_id") != cmd_id:
                continue
            stamp = f"{time.monotonic()-t0:6.1f}s"
            msg = (e.get("message") or "")[:140].replace("\n", " ")
            print(f"{stamp}  [{e.get('state')}] {msg}")
            if e.get("state") in TERMINAL:
                terminal = e["state"]
        if terminal:
            break
        time.sleep(0.3)

    print()
    elapsed = time.monotonic() - t0
    if terminal == "succeeded":
        print(f"RESULT: succeeded in {elapsed:.1f}s")
        sys.exit(0)
    elif terminal:
        print(f"RESULT: {terminal} after {elapsed:.1f}s")
        sys.exit(1)
    else:
        print(f"RESULT: TIMEOUT after {args.timeout}s")
        sys.exit(2)


if __name__ == "__main__":
    main()
