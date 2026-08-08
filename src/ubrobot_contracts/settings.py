"""Centralized runtime configuration for Operator Console and Robot Edge.

Replaces scattered ``os.environ.get("UBROBOT_*", default)`` reads with typed,
validated settings. The two services (console and edge) are independent
processes with distinct namespaces, so they get separate settings classes with
their own ``env_prefix``.

- ``ConsoleSettings`` reads ``UBROBOT_CHAT_*`` / ``UBROBOT_VOICE_*`` /
  ``UBROBOT_QWEN_*`` / ``UBROBOT_MOCK_*`` / ``UBROBOT_EDGE_*`` (console side
  backend connection).
- ``EdgeSettings`` reads ``UBROBOT_EDGE_*`` (edge service side).

These classes are pure Python (pydantic-settings only) so they are unit-testable
on a workstation. Nothing here imports ROS or hardware SDKs.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


class ConsoleSettings(BaseSettings):
    """Operator Console runtime configuration.

    Reads ``UBROBOT_CHAT_*``, ``UBROBOT_VOICE_*``, ``UBROBOT_QWEN_*``,
    ``UBROBOT_MOCK_*`` and the console-side ``UBROBOT_EDGE_*`` backend vars.
    """

    model_config = SettingsConfigDict(
        env_prefix="UBROBOT_CHAT_",
        env_file=None,
        extra="ignore",
        case_sensitive=False,
    )

    # Service
    host: str = "0.0.0.0"
    port: int = 7863
    log_level: str = "INFO"
    backend: str = "cortex"
    media: bool = True
    tls: bool = True
    shutdown_token: str = ""

    # Voice. voice_provider reads UBROBOT_VOICE_PROVIDER (no _CHAT_ segment);
    # qwen fields read UBROBOT_QWEN_*.
    voice_provider: str = Field(default="off", validation_alias="UBROBOT_VOICE_PROVIDER")
    qwen_model: str = Field(default="qwen3.5-omni-plus-realtime", validation_alias="UBROBOT_QWEN_REALTIME_MODEL")
    qwen_voice: str = Field(default="Tina", validation_alias="UBROBOT_QWEN_REALTIME_VOICE")
    qwen_region: str = Field(default="cn-beijing", validation_alias="UBROBOT_QWEN_REALTIME_REGION")
    qwen_proxy: str = Field(default="direct", validation_alias="UBROBOT_QWEN_REALTIME_PROXY")
    qwen_session_timeout_sec: float = Field(
        default=1800.0, validation_alias="UBROBOT_QWEN_REALTIME_SESSION_TIMEOUT_SEC"
    )

    # Mock backend timing (cortex-mock only). Read UBROBOT_MOCK_*.
    mock_nav_duration_sec: float = Field(default=4.0, validation_alias="UBROBOT_MOCK_NAV_DURATION_SEC")
    mock_reply_delay_sec: float = Field(default=0.3, validation_alias="UBROBOT_MOCK_REPLY_DELAY_SEC")

    # Edge backend connection (console side). These read UBROBOT_EDGE_*
    # (shared namespace), not UBROBOT_CHAT_EDGE_*.
    edge_url: str = Field(default="http://127.0.0.1:8780", validation_alias="UBROBOT_EDGE_URL")
    edge_operator_id: str = Field(default="operator", validation_alias="UBROBOT_EDGE_OPERATOR_ID")
    edge_token: str = Field(default="", validation_alias="UBROBOT_EDGE_TOKEN")
    edge_token_file: str = Field(default="", validation_alias="UBROBOT_EDGE_TOKEN_FILE")
    edge_local_hardware_permitted: bool = Field(
        default=False, validation_alias="UBROBOT_EDGE_LOCAL_HARDWARE_PERMITTED"
    )

    # DashScope (voice + VLM). Read DASHSCOPE_*.
    dashscope_api_key: str = Field(default="", validation_alias="DASHSCOPE_API_KEY")
    dashscope_workspace_id: str = Field(default="", validation_alias="DASHSCOPE_WORKSPACE_ID")

    @field_validator("backend")
    @classmethod
    def _backend_allowed(cls, v: str) -> str:
        value = v.strip().lower()
        if value not in {"cortex", "cortex-mock", "robot-edge", "legacy", "injected"}:
            raise ValueError(
                f"UBROBOT_CHAT_BACKEND must be one of cortex, cortex-mock, "
                f"robot-edge, legacy; got {value!r}"
            )
        return value

    @field_validator("voice_provider")
    @classmethod
    def _voice_allowed(cls, v: str) -> str:
        value = v.strip().lower()
        if value not in {"off", "disabled", "mock", "qwen"}:
            raise ValueError(
                f"UBROBOT_VOICE_PROVIDER must be off/mock/qwen; got {value!r}"
            )
        return value

    @field_validator("port")
    @classmethod
    def _port_in_range(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError(f"port must be 1..65535; got {v}")
        return v


class EdgeSettings(BaseSettings):
    """Robot Edge service runtime configuration.

    Reads ``UBROBOT_EDGE_*`` and ``UBROBOT_PLATFORM``.
    """

    model_config = SettingsConfigDict(
        env_prefix="UBROBOT_EDGE_",
        env_file=None,
        extra="ignore",
        case_sensitive=False,
    )

    # Service
    mode: Literal["fixture", "hardware"] = "fixture"
    host: str = "127.0.0.1"
    port: int = 8780
    log_level: str = "info"
    # Reads UBROBOT_PLATFORM (no _EDGE_ segment); shared across console+edge.
    platform: str = Field(default="", validation_alias="UBROBOT_PLATFORM")

    # Authority + safety
    hardware_authority: bool = False
    estop_enabled: bool = False
    estop_exempted: bool = False
    estop_chip: str = ""
    estop_line: str = ""
    estop_line_name: str = "ubrobot-estop"
    estop_debounce_sec: float = 0.02

    # Auth + replay protection
    tokens_file: str = ""
    request_max_age_sec: int = 300
    nonce_ttl_sec: int = 600

    # Fixture mode
    fixture_step_delay_sec: float = 0.0

    # Deployment paths (robot side)
    safety_checklist: str = ""

    @field_validator("mode")
    @classmethod
    def _mode_allowed(cls, v: str) -> str:
        if v not in {"fixture", "hardware"}:
            raise ValueError(f"UBROBOT_EDGE_MODE must be fixture/hardware; got {v!r}")
        return v

    @field_validator("port")
    @classmethod
    def _port_in_range(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError(f"port must be 1..65535; got {v}")
        return v

    @field_validator("request_max_age_sec", "nonce_ttl_sec")
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("request_max_age_sec / nonce_ttl_sec must be positive")
        return v


@lru_cache(maxsize=1)
def console_settings() -> ConsoleSettings:
    """Cached default ConsoleSettings read once from the environment.

    ``pipeline.py`` re-reads the backend on construction; callers that need a
    fresh view can construct ``ConsoleSettings()`` directly. The cache keeps
    import-time reads cheap while allowing explicit re-instantiation.
    """
    return ConsoleSettings()


@lru_cache(maxsize=1)
def edge_settings() -> EdgeSettings:
    """Cached default EdgeSettings read once from the environment."""
    return EdgeSettings()
