#!/usr/bin/env python3
"""Minimal HTTP -> HTTPS pass-through for OpenAI-compatible planner APIs.

EMOS ``GenericHTTPClient`` only builds ``http://host:port`` URLs, while most
hosted OpenAI-compatible providers are HTTPS-only. This relay listens on
plain HTTP (intended for localhost/container use) and forwards ``/v1/*``
requests to the configured HTTPS upstream, preserving method, body, and the
``Authorization`` header supplied by the client.

Configuration is via environment variables only; no credentials are stored
here — the API key travels in the client's Authorization header:

- ``PLANNER_RELAY_HOST`` (default ``0.0.0.0``)
- ``PLANNER_RELAY_PORT`` (default ``18081``)
- ``PLANNER_UPSTREAM_URL`` (required, e.g. ``https://api.deepseek.com``)
- ``PLANNER_UPSTREAM_TIMEOUT_SEC`` (default ``60``)
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
import ssl
import urllib.error
import urllib.request

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}


def make_handler(upstream: str, timeout_sec: float):
    class RelayHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _forward(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else None
            headers = {
                key: value
                for key, value in self.headers.items()
                if key.lower() not in HOP_BY_HOP_HEADERS
            }
            request = urllib.request.Request(
                upstream.rstrip("/") + self.path,
                data=body,
                headers=headers,
                method=self.command,
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=timeout_sec, context=ssl.create_default_context()
                ) as response:
                    payload = response.read()
                    status = response.status
                    content_type = response.headers.get(
                        "Content-Type", "application/json"
                    )
            except urllib.error.HTTPError as exc:
                payload = exc.read()
                status = exc.code
                content_type = exc.headers.get("Content-Type", "application/json")
            except (urllib.error.URLError, TimeoutError, ssl.SSLError) as exc:
                payload = ("{\"error\": \"upstream unreachable: %s\"}" % exc).encode(
                    "utf-8"
                )
                status = 502
                content_type = "application/json"

            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            # Never log bodies or headers: prompts and credentials must not
            # land in container logs.
            print(
                f"[planner-relay] {self.command} {self.path} -> {status}",
                flush=True,
            )

        do_GET = _forward  # noqa: N815 - stdlib handler API
        do_POST = _forward  # noqa: N815

        def log_message(self, format: str, *args) -> None:
            pass

    return RelayHandler


def main() -> int:
    upstream = os.environ.get("PLANNER_UPSTREAM_URL", "").strip()
    if not upstream.startswith("https://"):
        raise SystemExit(
            "PLANNER_UPSTREAM_URL must be an https:// URL, e.g. "
            "https://api.deepseek.com"
        )
    host = os.environ.get("PLANNER_RELAY_HOST", "0.0.0.0")
    port = int(os.environ.get("PLANNER_RELAY_PORT", "18081"))
    timeout_sec = float(os.environ.get("PLANNER_UPSTREAM_TIMEOUT_SEC", "60"))
    server = ThreadingHTTPServer((host, port), make_handler(upstream, timeout_sec))
    print(f"[planner-relay] {host}:{port} -> {upstream}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
