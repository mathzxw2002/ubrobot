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
        self.assertRegex(source, r"max_planning_steps\s*=\s*[1-9]")
        self.assertRegex(source, r"max_execution_steps\s*=\s*[1-9]")

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
