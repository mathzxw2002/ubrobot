"""Shared transport contracts between Operator Console and Robot Edge.

This package contains only Pydantic models, enums, and constants.
It must never import ROS, hardware SDKs, or framework-specific code.
"""

PROTOCOL_VERSION = "1.0"
