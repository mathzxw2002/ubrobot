from ubrobot_base import UBRobotBase
import random

class UBVisionRobot(UBRobotBase):
    """
    A robot with visual perception capabilities, inheriting from UBRobotBase.
    """

    def __init__(self, name: str):
        """
        Initializes the VisionRobot.
        """
        super().__init__(name)
        print(f"VisionRobot {self.name} is ready.")

    def perceive_objects(self) -> list[str]:
        print(f"{self.name} perceived the following objects: ")
        return "perceived_objects"

def demo():
    """
    Demonstrates the functionality of the VisionRobot.
    """
    print("--- Starting VisionRobot Demo ---")
    # Create an instance of VisionRobot
    vision_bot = UBVisionRobot("Wall-E")

    # Use methods from the base class
    vision_bot.speak("Hello, I am ready to explore!")
    vision_bot.move_forward(2.5)
    vision_bot.turn(90)

    # Use the new method from VisionRobot
    vision_bot.perceive_objects()

    vision_bot.speak("Exploration complete.")
    print("--- VisionRobot Demo Finished ---")

if __name__ == "__main__":
    demo()