from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class CortexApiContractTest(unittest.TestCase):
    def test_image_pins_the_verified_emos_base_digest(self):
        dockerfile = (ROOT / "deploy/emos/Dockerfile").read_text(encoding="utf-8")
        self.assertIn(
            "ARG EMOS_BASE_IMAGE=ghcr.io/automatika-robotics/emos@sha256:"
            "8ee294cffd187328ac3c2776e3389d8d93ad0bc7479e0dac284ae3d095e90f41",
            dockerfile,
        )

    def test_image_verifies_required_cortex_symbols(self):
        dockerfile = (ROOT / "deploy/emos/Dockerfile").read_text(encoding="utf-8")
        self.assertIn("COPY deploy/emos/verify_cortex_api.py", dockerfile)
        self.assertIn(
            "RUN /ros_entrypoint.sh python3 /opt/ubrobot/verify_cortex_api.py",
            dockerfile,
        )

    def test_probe_requires_cortex_action_discovery_surface(self):
        probe = (ROOT / "deploy/emos/verify_cortex_api.py").read_text(
            encoding="utf-8"
        )
        for token in ("Cortex", "CortexConfig", "Action", "Launcher"):
            self.assertIn(token, probe)


if __name__ == "__main__":
    unittest.main()
