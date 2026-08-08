"""Re-export of the pure motion-authority tracker (no ROS imports).

``AuthorityTracker`` and its thresholds moved to
``ubrobot_contracts.motion_authority`` (refactor Task 1) so the pure-Python
layer (Robot Edge arbiter, workstation tests) never depends on a ROS ament
package. This module keeps the historical import path working for the ROS
grasp server:

    from ubrobot_manipulation.authority import AuthorityTracker

Importing this module does not require ROS or any vendor SDK.
"""

from ubrobot_contracts.motion_authority import (
    AuthorityTracker,
    CMD_VEL_EPSILON,
    CMD_VEL_WINDOW_SEC,
    LEASE_MAX_AGE_SEC,
)

__all__ = [
    "AuthorityTracker",
    "LEASE_MAX_AGE_SEC",
    "CMD_VEL_WINDOW_SEC",
    "CMD_VEL_EPSILON",
]
