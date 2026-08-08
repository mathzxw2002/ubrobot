"""Static regression tests guarding against hardcoded credentials in source.

These tests scan tracked source/config files for the known leaked DashScope
key and for generic high-entropy ``sk-...`` secrets, and assert that the
credential-consuming modules read the key from ``DASHSCOPE_API_KEY`` instead
of embedding it. They run on a workstation without any SDK installed.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Keys that leaked through git history must never reappear in the working
# tree. Both were retired from the running services.
LEAKED_DASHSCOPE_KEYS = (
    # ec3ee21 "Resolve robot code merge conflicts" (vlm.py / tts.py)
    "sk-78b8ea9b14b944d0a2240408b8c766dd",
    # Pre-CI startup scripts (ubrobot_startup.sh, removed in ca2111f)
    "sk-479fdd23120c4201bff35a107883c7c3",
)

# Any DashScope-style key (sk- + 32 hex chars) counts as a live secret.
SECRET_RE = re.compile(r"sk-[0-9a-fA-F]{32}")

# Files that are allowed to be absent or to reference the env var name only.
_SCAN_PATTERNS = ("*.py", "*.sh", "*.ps1", "*.yaml", "*.yml", "*.json")


def _tracked_source_files() -> list[Path]:
    """All tracked files under src/, deploy/, scripts/ and examples/.

    The security test module itself is excluded: it intentionally embeds the
    retired keys as literals to prove they never reappear in real source.
    """
    result: list[Path] = []
    for sub in ("src", "deploy", "scripts", "examples"):
        base = ROOT / sub
        if not base.is_dir():
            continue
        for pattern in _SCAN_PATTERNS:
            result.extend(base.rglob(pattern))
    return [p for p in result if "__pycache__" not in p.parts]


class HardcodedSecretScanTest(unittest.TestCase):
    def test_leaked_keys_are_not_in_source(self) -> None:
        hits = []
        for path in _tracked_source_files():
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for key in LEAKED_DASHSCOPE_KEYS:
                if key in content:
                    hits.append(f"{path}:{key}")
        self.assertEqual(hits, [], f"leaked DashScope keys found in: {hits}")

    def test_no_high_entropy_sk_secret_in_source(self) -> None:
        hits = []
        for path in _tracked_source_files():
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for match in SECRET_RE.findall(content):
                hits.append(f"{path}:{match}")
        self.assertEqual(hits, [], f"high-entropy sk- secrets found in: {hits}")

    def test_vlm_reads_key_from_env(self) -> None:
        source = (ROOT / "src/ubrobot/robots/vlm.py").read_text(encoding="utf-8")
        self.assertIn("DASHSCOPE_API_KEY", source)
        self.assertIn("os.environ", source)
        for key in LEAKED_DASHSCOPE_KEYS:
            self.assertNotIn(key, source)

    def test_tts_reads_key_from_env(self) -> None:
        source = (ROOT / "src/ubrobot/robots/tts.py").read_text(encoding="utf-8")
        self.assertIn("DASHSCOPE_API_KEY", source)
        self.assertIn("os.environ", source)
        for key in LEAKED_DASHSCOPE_KEYS:
            self.assertNotIn(key, source)


if __name__ == "__main__":
    unittest.main()
