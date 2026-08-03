"""ROS context management for Robot Edge (read-only, M6).

The real implementation imports ``rclpy`` only inside the factory call, never
at module import time. This keeps workstation tests and fixture mode free of
any ROS dependency.
"""

from __future__ import annotations

from array import array
import threading
import time
from typing import Any, Protocol


class RosGraph(Protocol):
    """Read-only view of a ROS 2 graph.

    Implementations must return only JSON-serializable values (no ROS
    messages, no SDK objects, no callbacks cross this boundary).
    """

    def has_topic(self, topic: str) -> bool: ...

    def read_topic(self, topic: str) -> dict[str, Any] | None:
        """JSON-safe snapshot of the latest message, or None when no message
        is available within the read window."""

    def has_action_server(self, action_name: str) -> bool: ...

    def shutdown(self) -> None: ...


class RosNodeGraph:
    """Real ``rclpy``-backed read-only graph.

    Constructed only by :func:`create_ros_context` in hardware mode. All
    operations are read-only: subscriptions observe, nothing publishes, no
    service calls, no action clients, no control topics.
    """

    def __init__(self, node: Any) -> None:
        # ``node`` is a lazily-imported rclpy node instance.
        self._node = node
        self._topics: dict[str, set[str]] | None = None
        # rclpy.spin_once on the global executor is not thread-safe; multiple
        # concurrent callers (telemetry snapshots, command backend) race and
        # raise "Executor is already spinning". Serialize all global-spin use.
        self.spin_lock = threading.RLock()

    def _refresh(self) -> None:
        with self.spin_lock:
            topics: dict[str, set[str]] = {}
            for name, types in self._node.get_topic_names_and_types():
                topics[name] = set(types)
            self._topics = topics

    def has_topic(self, topic: str) -> bool:
        # Always refresh: the ROS graph changes as nodes come and go (e.g. a
        # sensor node that was still starting when the Edge booted). A stale
        # cache would report a channel as unavailable forever.
        self._refresh()
        assert self._topics is not None
        return topic in self._topics

    def read_topic(self, topic: str) -> dict[str, Any] | None:
        import rclpy  # noqa: PLC0415 - hardware-only import

        self._refresh()
        assert self._topics is not None
        types = self._topics.get(topic)
        if not types:
            return None
        message_type = _import_message_type(sorted(types)[0])
        received: list[Any] = []

        def on_message(msg: Any) -> None:
            received.append(msg)

        sub = self._node.create_subscription(
            message_type,
            topic,
            on_message,
            qos_profile=10,  # sensor data, best-effort, read-only
        )
        try:
            with self.spin_lock:
                deadline = time.monotonic() + 1.0
                while not received and time.monotonic() < deadline:
                    rclpy.spin_once(self._node, timeout_sec=0.1)
            if not received:
                return None
            return _json_safe(received[0])
        finally:
            self._node.destroy_subscription(sub)

    def has_action_server(self, action_name: str) -> bool:
        """A ROS 2 action server exposes ``<action>/_action/status`` etc."""
        if self._topics is None:
            self._refresh()
        assert self._topics is not None
        base = action_name.rstrip("/")
        return any(
            topic == f"{base}/_action/status" or topic == f"{base}/_action/send_goal"
            for topic in self._topics
        )

    def shutdown(self) -> None:
        self._topics = None


def _import_message_type(type_name: str) -> Any:
    """Import a ROS message class from a graph type name.

    Accepts ``pkg/msg/Type`` (ROS 2 naming). Returns the message class, or
    raises ValueError for unsupported shapes.
    """
    pkg, _, rest = type_name.partition("/")
    parts = rest.split("/")
    if len(parts) != 2 or parts[0] != "msg":
        raise ValueError(f"unsupported ROS type name: {type_name}")
    module = __import__(f"{pkg}.msg", fromlist=[parts[1]])
    return getattr(module, parts[1])


def _message_field_names(message: Any) -> list[str]:
    """Public field names of a ROS 2 generated Python message.

    rclpy-generated classes (e.g. ``nav_msgs.msg.Odometry``) declare slots
    with private names (``_pose``, ``_twist``, ...) and expose public
    read-only properties without an instance ``__dict__``. Strip the leading
    underscore so ``_json_safe`` emits the public field names; fall back to
    the instance dict for plain objects.
    """
    slots = getattr(message, "__slots__", ()) or ()
    if slots:
        return [str(name).lstrip("_") for name in slots if str(name)]
    return list(getattr(message, "__dict__", {}).keys())


def _json_safe(message: Any, *, _depth: int = 0) -> dict[str, Any]:
    """Extract a JSON-safe subset of a ROS message.

    Scalar fields are kept; byte fields are reduced to their size (never the
    raw frame); nested messages/arrays are converted recursively up to a
    bounded depth. SDK-like objects that are neither ROS messages nor plain
    containers are dropped.
    """
    if message is None or _depth > 4:
        return {}
    if isinstance(message, dict):
        result: dict[str, Any] = {}
        for key, value in message.items():
            if isinstance(value, dict):
                result[str(key)] = _json_safe(value, _depth=_depth + 1)
            elif isinstance(value, list):
                result[str(key)] = _json_safe_list(value, _depth=_depth + 1)
            elif isinstance(value, (str, int, float, bool)):
                result[str(key)] = value
        return result
    data: dict[str, Any] = {}
    for name in _message_field_names(message):
        try:
            value = getattr(message, name)
        except Exception:
            continue
        if isinstance(value, (str, int, float, bool)):
            data[name] = value
        elif isinstance(value, (list, array)):
            data[name] = _json_safe_list(value, _depth=_depth + 1)
        elif isinstance(value, dict):
            data[name] = _json_safe(value, _depth=_depth + 1)
        elif isinstance(value, (bytes, bytearray)):
            data[name] = len(value)  # size only, never the raw frame
        elif hasattr(value, "__slots__") or hasattr(value, "__dict__"):
            data[name] = _json_safe(value, _depth=_depth + 1)
    return data


def _json_safe_list(values: Any, *, _depth: int) -> list[Any]:
    """Convert a ROS array to JSON-safe scalars/containers.

    rclpy float64[]/int[] fields are ``array.array`` instances, not Python
    lists; normalize them first so the scalars survive serialization.
    """
    if _depth > 4:
        return []
    if isinstance(values, array):
        values = values.tolist()
    result: list[Any] = []
    for value in values:
        if isinstance(value, (str, int, float, bool)):
            result.append(value)
        elif isinstance(value, (bytes, bytearray)):
            result.append(len(value))
        elif isinstance(value, (list, array)):
            result.append(_json_safe_list(value, _depth=_depth + 1))
        elif isinstance(value, dict):
            result.append(_json_safe(value, _depth=_depth + 1))
        elif hasattr(value, "__slots__"):
            result.append(_json_safe(value, _depth=_depth + 1))
    return result


def create_ros_context(*, execution_mode: str) -> RosGraph | None:
    """Create a read-only ROS graph, or None outside hardware mode.

    ``rclpy`` is imported only when hardware mode is requested. Any other
    mode returns None without touching the ROS stack.
    """
    if execution_mode != "hardware":
        return None
    import rclpy  # noqa: PLC0415 - hardware-only import
    from rclpy.node import Node  # noqa: PLC0415

    if not rclpy.ok():
        rclpy.init(args=[])
    node = Node("robot_edge_readonly")
    return RosNodeGraph(node)
