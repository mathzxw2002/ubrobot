
class UBRobotBase:
    """A base class for robots, providing basic action capabilities."""

    def __init__(self, name: str):
        """Initializes the robot with a name."""
        self.name = name
        print(f"Robot {self.name} initialized.")

    def move_forward(self, distance: float):
        """Moves the robot forward by a specified distance."""
        print(f"{self.name} is moving forward by {distance} meters.")

    def turn(self, angle: float):
        """Turns the robot by a specified angle in degrees."""
        print(f"{self.name} is turning by {angle} degrees.")

    def speak(self, message: str):
        """Makes the robot speak a message."""
        print(f"{self.name} says: {message}")
