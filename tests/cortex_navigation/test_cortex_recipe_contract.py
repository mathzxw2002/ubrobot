import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "deploy/emos/recipes/cortex_navigation/recipe.py"
OVERRIDE = ROOT / "deploy/emos/compose.cortex-navigation.yaml"
DOCKERFILE = ROOT / "deploy/emos/Dockerfile"
SUPERVISOR = ROOT / "deploy/emos/start-stack.sh"


def assigned_string(source, name):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return ast.literal_eval(node.value)
    raise AssertionError(f"{name} assignment not found")


class CortexRecipeContractTest(unittest.TestCase):
    def test_recipe_uses_bounded_cortex_planning(self):
        source = RECIPE.read_text(encoding="utf-8")
        self.assertIn("from agents.ros import Launcher", source)
        self.assertIn("Cortex(", source)
        self.assertIn("CortexConfig(", source)
        # Bounds stay small by default and are env-tunable, never unlimited.
        self.assertRegex(
            source,
            r'max_planning_steps=int\(os\.environ\.get\('
            r'"CORTEX_MAX_PLANNING_STEPS",\s*"[1-9]',
        )
        self.assertRegex(
            source,
            r'max_execution_steps=int\(\s*os\.environ\.get\('
            r'"CORTEX_MAX_EXECUTION_STEPS",\s*"[1-9]',
        )

    def test_cortex_tuning_comes_from_environment_with_defaults(self):
        source = RECIPE.read_text(encoding="utf-8")
        for name, default in (
            ("CORTEX_MONITORING_INTERVAL_SEC", "0.5"),
            ("CORTEX_TEMPERATURE", "0.1"),
            ("CORTEX_MAX_NEW_TOKENS", "600"),
        ):
            self.assertIn(f'os.environ.get("{name}", "{default}")', source)
        compose = OVERRIDE.read_text(encoding="utf-8")
        self.assertIn("CORTEX_MONITORING_INTERVAL_SEC", compose)

    def test_recipe_discovers_only_the_semantic_navigation_action(self):
        source = RECIPE.read_text(encoding="utf-8")
        self.assertIn("NavigationCapabilityProxy", source)
        self.assertIn("NavigateToObject", source)
        self.assertIn('"/ubrobot/navigation/navigate_to_object"', source)
        self.assertIn("self._managed_components =", source)
        self.assertIn('action_servers_components=[]', source)

    def test_navigation_tool_description_stays_semantic(self):
        description = assigned_string(
            RECIPE.read_text(encoding="utf-8"),
            "NAVIGATION_TOOL_DESCRIPTION",
        ).lower()
        for phrase in (
            "visually detectable object label",
            "can be cancelled",
            "sensors",
            "detection",
            "localization",
        ):
            self.assertIn(phrase, description)
        for forbidden in ("/cmd_vel", "serial", "torque", "motor id", "/dev/"):
            self.assertNotIn(forbidden, description)

    def test_grasp_tool_is_semantic_and_gated(self):
        source = RECIPE.read_text(encoding="utf-8")
        self.assertIn(
            'GRASP_TOOL_NAME = "send_goal_to__ubrobot_manipulation_grasp_object"',
            source,
        )
        self.assertIn("SemanticCapabilityProxy(", source)
        self.assertIn("grasp_exposure_enabled", source)
        description = assigned_string(source, "GRASP_TOOL_DESCRIPTION").lower()
        for phrase in (
            "visually detectable object label",
            "robot arm",
            "can be cancelled",
            "never moves the mobile base",
            "may fail",
        ):
            self.assertIn(phrase, description)
        for forbidden in ("/cmd_vel", "serial", "torque", "motor id", "/dev/"):
            self.assertNotIn(forbidden, description)

    def test_grasp_exposure_passthrough_in_compose(self):
        source = OVERRIDE.read_text(encoding="utf-8")
        self.assertIn("CORTEX_ENABLE_GRASP: ${CORTEX_ENABLE_GRASP:-false}", source)

    def test_planner_model_configuration_comes_from_environment(self):
        source = RECIPE.read_text(encoding="utf-8")
        for name in (
            "CORTEX_MODEL_HOST",
            "CORTEX_MODEL_PORT",
            "CORTEX_MODEL_CHECKPOINT",
            "CORTEX_MODEL_API_KEY",
        ):
            self.assertIn(f'os.environ.get("{name}"', source)
        self.assertNotIn("sk-", source)

    def test_image_keeps_both_recipes(self):
        dockerfile = DOCKERFILE.read_text(encoding="utf-8")
        self.assertIn("recipes/vision_depth_follower/recipe.py", dockerfile)
        self.assertIn("recipes/cortex_navigation/recipe.py", dockerfile)

    def test_supervisor_seeds_selected_recipe_without_overwrite(self):
        source = SUPERVISOR.read_text(encoding="utf-8")
        self.assertIn("/opt/ubrobot/recipes/${RECIPE_RELATIVE}", source)
        self.assertIn('if [ ! -f "${RECIPE}" ]', source)
        self.assertIn("cortex_navigation_bringup.launch.py", source)

    def test_compose_override_selects_recipe_without_device_access(self):
        source = OVERRIDE.read_text(encoding="utf-8")
        self.assertIn("EMOS_RECIPE: /emos/recipes/cortex_navigation/recipe.py", source)
        for forbidden in ("devices:", "/dev/", "privileged:", "command:", "entrypoint:"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
