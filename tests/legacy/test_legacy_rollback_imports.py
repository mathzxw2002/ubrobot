"""Regression tests for the isolated legacy hardware-direct rollback code.

Verifies (without any robot hardware or vendor SDKs installed):

- importing ``ubrobot.robots.unitree_go2_robot`` does NOT require
  unitree_sdk2py at module load (lazy import) and emits DeprecationWarning;
- ``ubrobot.robots.ubrobot`` (Go2Manager) no longer connects hardware in
  ``__init__`` — the LeKiwi base connection moved to an explicit
  ``connect_base()`` (static source assertion; the module's heavy ML deps like
  open3d are not installed on a bare workstation);
- ``chat_ui.pipeline._LegacyBackend`` surfaces a clear error when hardware
  init fails instead of a bare crash.
"""

from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


class LegacyImportSafetyTest(unittest.TestCase):
    def test_go2_robot_lazy_imports_sdk_and_warns(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import ubrobot.robots.unitree_go2_robot as go2_mod  # noqa: F401

        self.assertTrue(
            any(issubclass(w.category, DeprecationWarning) for w in caught),
            "expected a DeprecationWarning on import",
        )
        source = Path(go2_mod.__file__).read_text(encoding="utf-8")
        # unitree_sdk2py must be imported lazily inside __init__, not at top.
        self.assertNotIn("import unitree_sdk2py", source.split("def __init__")[0])

    def test_go2manager_init_does_not_connect_base(self) -> None:
        # Static guard: constructing Go2Manager must not attach to hardware.
        # (Full import needs open3d/torch, so we assert on source.)
        source = (
            ROOT / "src/ubrobot/robots/ubrobot.py"
        ).read_text(encoding="utf-8")
        self.assertIn("def connect_base(self) -> None:", source)
        # __init__ may construct LeKiwi (cheap object) but must not call
        # connect() on it.
        init_body = source.split("def __init__", 1)[1].split("def ", 1)[0]
        self.assertNotIn("lekiwi_base.connect()", init_body)
        self.assertNotIn("self._base_connected = True", init_body)

    def test_go2manager_uses_package_relative_thread_utils(self) -> None:
        source = (
            ROOT / "src/ubrobot/robots/ubrobot.py"
        ).read_text(encoding="utf-8")
        self.assertIn("from .thread_utils import ReadWriteLock", source)

    def test_legacy_backend_surfaces_clear_error_on_hardware_failure(self) -> None:
        import sys as _sys
        import types as _types
        import unittest.mock as mock

        from chat_ui.pipeline import _LegacyBackend

        # Inject a fake ubrobot.robots.ubrobot module whose Go2Manager raises
        # on construction. Avoids importing the real module (heavy ML deps).
        fake_module = _types.ModuleType("ubrobot.robots.ubrobot")
        fake_module.Go2Manager = mock.Mock(side_effect=RuntimeError("no hardware"))
        with mock.patch.dict(_sys.modules, {"ubrobot.robots.ubrobot": fake_module}):
            with self.assertRaises(RuntimeError) as raised:
                _LegacyBackend()
        self.assertIn("legacy backend failed", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
