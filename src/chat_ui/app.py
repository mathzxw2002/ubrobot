import asyncio
import json
import logging
import os
import queue
import secrets
import shutil
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import gradio as gr
import uvicorn
from fastapi import (
    FastAPI,
    Header,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel, Field

try:  # Package import for tests and `python -m chat_ui.app`.
    from .pipeline import ChatPipeline
    from .service_lifecycle import (
        PortInUseError,
        require_port_available,
        sanitized_capability_health,
        shutdown_pipeline,
    )
except ImportError:  # Script compatibility: `python src/chat_ui/app.py`.
    from pipeline import ChatPipeline
    from service_lifecycle import (
        PortInUseError,
        require_port_available,
        sanitized_capability_health,
        shutdown_pipeline,
    )


LOG_LEVEL = os.environ.get("UBROBOT_CHAT_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("ubrobot.operator_console")

VOICE_CLIENT_JS = (
    Path(__file__).with_name("voice_client.js").read_text(encoding="utf-8")
)

chat_pipeline = None


class OperatorInteractionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    source: str = Field(default="text", pattern=r"^[a-zA-Z0-9_.-]+$")
    correlation_id: str | None = Field(default=None, max_length=200)


def execute_operator_interaction(text, *, source="text", correlation_id=None):
    """Shared interaction entry point for Gradio, HTTP, and acceptance tests."""
    return chat_pipeline.request_interaction(
        text,
        source=source,
        correlation_id=correlation_id,
    )


def _interaction_payload(result):
    return {
        "turn": result.turn.to_dict(),
        "reply": result.reply,
        "dispatched": result.dispatched,
        "task_id": result.task_id,
    }


def _escape_cell(value):
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def _task_status_markdown(snapshot):
    tasks = snapshot["tasks"]
    active = tasks["active_task"]
    pending = tasks["pending_tasks"]
    if active is None:
        active_text = "空闲"
    else:
        active_text = f"`{active['status']}` · {_escape_cell(active['intent'])}"
    return (
        "### Task 状态\n\n"
        f"- 当前主任务：{active_text}\n"
        f"- 待处理任务：{len(pending)}\n"
        f"- Cortex 后端：`{chat_pipeline.backend_name}`"
    )


def _timeline_markdown(snapshot):
    events = snapshot["tasks"]["events"][-12:]
    if not events:
        return "### Task 时间线\n\n暂无任务事件。"
    rows = ["### Task 时间线", "", "| 时间 | 事件 | 信息 |", "|---|---|---|"]
    for event in reversed(events):
        timestamp = event["timestamp"][11:19]
        rows.append(
            f"| {timestamp} | `{_escape_cell(event['event_type'])}` | "
            f"{_escape_cell(event['message'])} |"
        )
    return "\n".join(rows)


_CHANNEL_TITLES = {
    "camera": "📷 相机 RGB",
    "depth": "🕳️ 深度相机",
    "odometry": "🧭 里程计",
    "joint_states": "⚙️ 关节状态",
    "navigation_lease": "🔑 导航租约",
    "capability_health": "🧩 能力健康",
}

# Labels for the meaningful telemetry fields; envelope/boilerplate keys
# (channel/state/available/source/timestamp) are intentionally skipped.
_FIELD_LABELS = {
    "width": "宽度",
    "height": "高度",
    "unit": "单位",
    "encoding": "编码",
    "frame_id": "坐标系",
    "calibrated": "已标定",
    "frame_matches_expected": "帧匹配",
    "kind": "类型",
    "distortion_model": "畸变模型",
    "x": "x",
    "y": "y",
    "yaw": "yaw",
    "vx": "线速度",
    "profile": "底盘配置",
    "motor_count": "电机数",
    "names": "关节",
    "positions": "位置",
    "velocities": "速度",
    "owner": "持有者",
    "lease_id": "租约 ID",
    "expires_at": "到期时间",
    "topic": "话题",
    "age_sec": "消息龄",
    "detail": "详情",
}


def _fmt_value(value) -> str:
    """Format one scalar for display (floats with two decimals)."""
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _telemetry_value_fields(value) -> dict:
    """Extract semantic fields, unwrapping the Robot Edge bridge envelope.

    The edge telemetry client publishes
    ``{"channel", "state", "available", "source", "value": {...}}``; the
    fixture adapters publish flat DTO dicts. Both shapes are supported.
    """
    if not isinstance(value, dict):
        return {}
    nested = value.get("value")
    if isinstance(nested, dict) and (
        "state" in value or "available" in value or "channel" in value
    ):
        return dict(nested)
    return dict(value)


def _telemetry_state_label(sample) -> str:
    if sample.get("disconnected"):
        return "断连"
    if sample.get("available"):
        return "陈旧" if sample.get("stale") else "正常"
    return "不可用"


def _channel_summary(channel: str, fields: dict) -> str:
    """One-line key-value summary for the overview table."""
    if channel in ("camera", "depth"):
        width, height = fields.get("width"), fields.get("height")
        # The console's own camera refresh publishes ``size`` as a [w, h]
        # list; use it when the edge's width/height are not present.
        if width is None and isinstance(fields.get("size"), (list, tuple)):
            size = fields["size"]
            if len(size) >= 2:
                width, height = size[0], size[1]
        parts = []
        if width is not None and height is not None:
            parts.append(f"{width}×{height}")
        if fields.get("encoding"):
            parts.append(str(fields["encoding"]))
        if channel == "depth" and fields.get("unit"):
            parts.append(str(fields["unit"]))
        calibrated = fields.get("calibrated")
        if isinstance(calibrated, bool):
            parts.append("已标定" if calibrated else "未标定")
        return " · ".join(parts) or "-"
    if channel == "odometry":
        parts = [
            f"{key}={_fmt_value(fields[key])}"
            for key in ("x", "y", "yaw", "vx")
            if key in fields and fields[key] is not None
        ]
        return ", ".join(parts) or "-"
    if channel == "joint_states":
        names = fields.get("names") or []
        positions = fields.get("positions") or []
        count = fields.get("motor_count")
        if count is None:
            count = len(names)
        if positions:
            brief = ", ".join(
                f"{name}:{_fmt_value(position)}"
                for name, position in zip(names[:4], positions[:4])
            )
            suffix = "…" if len(names) > 4 else ""
            return f"{count} 电机 [{brief}{suffix}]"
        return f"{count} 电机" if count else "-"
    if channel == "navigation_lease":
        owner = fields.get("owner")
        if owner:
            expires = fields.get("expires_at")
            extra = f" 至 {str(expires)[:19]}" if expires else ""
            return f"持有者={owner}{extra}"
        return "无"
    if channel == "capability_health":
        caps = fields.get("capabilities")
        if isinstance(caps, dict) and caps:
            parts = [
                f"{name}:{item.get('health', '?')}"
                for name, item in caps.items()
                if isinstance(item, dict)
            ]
            return " · ".join(parts[:5]) or "-"
        return "-"
    return "-"


def _channel_detail_lines(fields: dict) -> list[str]:
    """Detail lines for one channel's value dict (label: value)."""
    lines = []
    for key, label in _FIELD_LABELS.items():
        if key not in fields or fields[key] is None:
            continue
        value = fields[key]
        if isinstance(value, dict):
            continue
        if isinstance(value, (list, tuple)):
            if not value:
                continue
            rendered = ", ".join(_fmt_value(item) for item in value[:8])
            suffix = "…" if len(value) > 8 else ""
            lines.append(f"- {label}: {rendered}{suffix}")
        else:
            lines.append(f"- {label}: {_fmt_value(value)}")
    return lines


def _telemetry_markdown(snapshot):
    """Render robot/sensor telemetry as readable Markdown for the console.

    Shows one overview table (state + age + key values) and one detail
    block per channel with the actual sensor values (odometry, joint
    positions, camera metadata, lease owner, capability health). Values
    come only from the serialized telemetry hub; no SDK objects or
    credentials are ever rendered.
    """
    telemetry = snapshot["telemetry"]
    lines = [
        "### 机器人与 Capability 状态",
        "",
        "| 通道 | 状态 | 更新 | 关键值 |",
        "|---|---|---|---|",
    ]
    for channel, sample in telemetry.items():
        title = _CHANNEL_TITLES.get(channel, channel)
        state = _telemetry_state_label(sample)
        age = sample.get("age_sec")
        age_text = "—" if age is None else f"{age:.1f}s 前"
        fields = _telemetry_value_fields(sample.get("value"))
        summary = _channel_summary(channel, fields)
        lines.append(f"| {title} | {state} | {age_text} | {summary} |")

    lines.append("")
    for channel, sample in telemetry.items():
        title = _CHANNEL_TITLES.get(channel, channel)
        state = _telemetry_state_label(sample)
        age = sample.get("age_sec")
        age_text = "—" if age is None else f"{age:.1f}s 前"
        lines.append(f"**{title}** — `{state}`（{age_text}）")
        fields = _telemetry_value_fields(sample.get("value"))
        detail_lines = _channel_detail_lines(fields)
        if not detail_lines:
            detail = fields.get("detail")
            if detail:
                lines.append(f"- 详情: {detail}")
            else:
                lines.append("- （无详细数据）")
        else:
            lines.extend(detail_lines)
        lines.append("")
    return "\n".join(lines)


def _voice_status_markdown(snapshot):
    voice = snapshot["voice"]
    state_labels = {
        "idle": "未启动",
        "connecting": "连接中",
        "listening": "正在聆听",
        "thinking": "Cortex 处理中",
        "speaking": "正在播报（普通语音暂停）",
        "emergency_stopped": "已紧急停止",
        "error": "错误",
    }
    lines = [
        "### 实时语音",
        f"- Provider：`{_escape_cell(voice['provider'])}`",
        f"- 状态：**{state_labels.get(voice['state'], voice['state'])}**",
    ]
    transcript = voice.get("transcript_partial") or voice.get("transcript_final")
    if transcript:
        lines.append(f"- 转写：{_escape_cell(transcript)}")
    if voice.get("last_error"):
        lines.append(f"- 错误：{_escape_cell(voice['last_error'])}")
    return "\n".join(lines)


def _input_text_and_files(value):
    if value is None:
        return "", []
    if isinstance(value, str):
        return value, []
    if isinstance(value, dict):
        return str(value.get("text") or ""), list(value.get("files") or [])
    return str(getattr(value, "text", "") or ""), list(
        getattr(value, "files", []) or []
    )


def _file_path(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("path") or value.get("name")
    return getattr(value, "path", None) or getattr(value, "name", None)


def submit_operator_turn(value, history):
    """Handle one native Gradio text/microphone submission.

    Non-blocking: the interaction runs in a background thread so the Gradio
    UI keeps rendering the task timeline and telemetry while Cortex/Kompass
    execute (which can take ~60 s). The final reply is appended to the chat
    by the timer callback when the background thread finishes.
    """
    text, files = _input_text_and_files(value)
    source = "voice" if files else "text"
    if files:
        path = _file_path(files[0])
        if chat_pipeline.asr is None:
            text = f"{text} [ASR disabled: media off]".strip()
        elif path:
            text = f"{text}{chat_pipeline.asr.infer(path)}".strip()
    text = text.strip()
    current_history = list(history or [])
    if not text:
        return gr.update(value=None, interactive=True), current_history, "请输入指令。"

    category = chat_pipeline.interaction_runtime.classify(text).value
    logger.info("interaction received source=%s category=%s", source, category)
    current_history.append({"role": "user", "content": text})
    current_history.append({"role": "assistant", "content": "任务已提交，正在执行..."})

    def run() -> None:
        try:
            result = execute_operator_interaction(text, source=source)
            chat_pipeline.record_completed(text, result.reply)
            logger.info("background interaction completed category=%s", category)
        except Exception as exc:  # noqa: BLE001 - surface as chat reply
            # Business failures (e.g. Cortex ABORTED, hardware authority
            # disabled) are expected outcomes, not programming errors:
            # report the reason without a traceback.
            logger.info(
                "background interaction failed category=%s reason=%s",
                category,
                exc,
            )
            chat_pipeline.record_completed(text, f"执行失败：{exc}")

    threading.Thread(target=run, daemon=True, name="operator-submit").start()
    return (
        gr.update(value=None, interactive=True),
        current_history,
        f"最近交互：`{source}` / `{category}`（执行中）",
    )


def stop_operator_task():
    active = chat_pipeline.task_runtime.active_task()
    if active is None:
        logger.info("stop ignored: no active task")
        return "当前没有正在执行的主任务。"
    acknowledged = chat_pipeline.task_runtime.cancel_active()
    logger.info(
        "stop requested task_id=%s acknowledged=%s",
        active.task_id,
        acknowledged,
    )
    return "已请求取消当前任务。" if acknowledged else "取消请求已发送，等待确认。"


def start_voice_session():
    # Browser JavaScript obtains microphone permission and opens the PCM
    # WebSocket. The WebSocket endpoint starts the provider only after that,
    # avoiding a billable cloud session when permission is denied.
    snapshot = chat_pipeline.voice_runtime.snapshot().to_dict()
    return (
        _voice_status_markdown({"voice": snapshot}),
        "正在请求麦克风权限并建立实时语音会话。",
    )


def stop_voice_session():
    snapshot = chat_pipeline.voice_runtime.stop().to_dict()
    logger.info("voice session stopped provider=%s", snapshot["provider"])
    return _voice_status_markdown({"voice": snapshot}), "实时语音会话已结束。"


def emergency_stop_operator():
    acknowledged = chat_pipeline.voice_runtime.emergency_stop("operator-console")
    snapshot = chat_pipeline.voice_runtime.snapshot().to_dict()
    logger.warning("emergency stop requested acknowledged=%s", acknowledged)
    notice = "紧急停止已确认。" if acknowledged else "紧急停止已发出，后端尚未确认。"
    return _voice_status_markdown({"voice": snapshot}), notice


def _camera_view_markdown(image, snapshot) -> str:
    """Status line for the operator camera panel (robot-eye view, M8).

    Combines the frame presence (from ``/v1/camera/frame``) with the
    camera telemetry channel state (from ``/v1/telemetry/snapshot``):
    live frame -> 正常, connected but no frame yet -> 等待画面,
    edge unreachable / no channel data -> 不可用.
    """
    camera = (snapshot.get("telemetry") or {}).get("camera") or {}
    if image is not None:
        size = getattr(image, "size", None)
        if isinstance(size, (tuple, list)) and len(size) >= 2:
            resolution = f"{int(size[0])}×{int(size[1])}"
        else:
            resolution = "未知分辨率"
        lines = [f"**📷 相机实时画面** — 状态：**正常**（{resolution}，每 2 秒刷新）"]
    elif camera.get("disconnected") or not camera.get("available"):
        lines = ["**📷 相机状态：不可用** — Robot Edge 无画面数据（离线或未启动）。"]
    else:
        lines = ["**📷 相机状态：等待画面** — 相机已连接，尚未收到图像帧。"]
    age = camera.get("age_sec")
    if age is not None:
        lines.append(f"- 遥测更新：{age:.1f}s 前")
    return "\n".join(lines)


def _camera_panel_update(image, snapshot):
    """The three Gradio outputs of the live camera panel (one refresh tick)."""
    has_image = image is not None
    return (
        gr.update(value=image, visible=has_image),
        gr.update(
            value="暂无画面：相机离线或尚未收到帧。",
            visible=not has_image,
        ),
        _camera_view_markdown(image, snapshot),
    )


def refresh_camera_once():
    """Fast camera-only refresh (~2 s) for the live robot-eye view.

    The main 5 s timer keeps chat/timeline/telemetry updates; this timer
    only re-fetches the latest JPEG from Robot Edge so the operator view
    follows the robot without re-rendering the whole console.
    """
    nav_image, _manipulation_image = chat_pipeline.get_robot_observation()
    snapshot = chat_pipeline.operator_snapshot()
    return _camera_panel_update(nav_image, snapshot)


# ── cached markdown outputs (avoid DOM flicker when content is unchanged) ──
_last_markdown: dict[str, str] = {}


def _cached_markdown(key: str, content: str) -> gr.update:
    """Return gr.update(value=...) only when *content* differs from last tick."""
    if _last_markdown.get(key) == content:
        return gr.update()
    _last_markdown[key] = content
    return gr.update(value=content)


def refresh_chat_once(history=None):
    """Fast chat-only refresh (~1 s): replace interaction placeholders.

    Only touches the Chatbot component; the 5 s console timer handles
    task/timeline/telemetry/voice separately.  Returns gr.update() (not
    None) so Gradio never clears the chat history.
    """
    completed = chat_pipeline.take_completed()
    if not completed:
        return gr.update()
    current_history = list(history or [])
    for _, reply in completed:
        if (
            current_history
            and current_history[-1].get("role") == "assistant"
            and current_history[-1].get("content") == "任务已提交，正在执行..."
        ):
            current_history[-1] = {"role": "assistant", "content": reply}
        else:
            current_history.append({"role": "assistant", "content": reply})
    return gr.update(value=current_history)


def refresh_console_once():
    """5 s console refresh: task status, timeline, telemetry, voice.

    Does NOT touch the chat history (that has its own 1 s timer).
    Uses content caching so Markdown panels only re-render on change.
    """
    snapshot = chat_pipeline.operator_snapshot()
    return (
        _cached_markdown("task", _task_status_markdown(snapshot)),
        _cached_markdown("timeline", _timeline_markdown(snapshot)),
        _cached_markdown("telemetry", _telemetry_markdown(snapshot)),
        _cached_markdown("voice", _voice_status_markdown(snapshot)),
    )


def operator_update_once(history=None):
    """Legacy combined refresh — kept for backward compatibility.

    New code should use refresh_camera_once + refresh_chat_once +
    refresh_console_once instead.
    """
    nav_image, manipulation_image = chat_pipeline.get_robot_observation()
    is_manipulate_valid = (
        manipulation_image is not None and getattr(manipulation_image, "size", 1) != 0
    )
    snapshot = chat_pipeline.operator_snapshot()
    nav_image_update, nav_placeholder_update, camera_status_update = (
        _camera_panel_update(nav_image, snapshot)
    )
    history_updated = gr.update()
    completed = chat_pipeline.take_completed()
    if completed:
        current_history = list(history or [])
        for _, reply in completed:
            if (
                current_history
                and current_history[-1].get("role") == "assistant"
                and current_history[-1].get("content") == "任务已提交，正在执行..."
            ):
                current_history[-1] = {"role": "assistant", "content": reply}
            else:
                current_history.append({"role": "assistant", "content": reply})
        history_updated = gr.update(value=current_history)
    return (
        nav_image_update,
        nav_placeholder_update,
        camera_status_update,
        gr.update(value=manipulation_image, visible=is_manipulate_valid),
        _task_status_markdown(snapshot),
        _timeline_markdown(snapshot),
        _telemetry_markdown(snapshot),
        _voice_status_markdown(snapshot),
        history_updated,
    )


# Compatibility for callers importing the historical refresh callback.
gradio_planning_txt_update = operator_update_once


def create_gradio():
    voice_head = f"<script>({VOICE_CLIENT_JS})();</script>"
    with gr.Blocks(
        title="UBRobot ChatUI",
        head=voice_head,
        analytics_enabled=False,
    ) as demo:
        gr.Markdown(
            """
            <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
            UBRobot Operator Console
            </div>
            """
        )
        mock_mode = chat_pipeline.backend_name == "cortex-mock"
        gr.Markdown(
            """
            <div style="background:#8b1e1e;color:white;border:2px solid #ff6b6b;
                        padding:12px;text-align:center;font-size:20px;font-weight:800;
                        border-radius:8px;margin-bottom:14px;">
            MOCK / NO HARDWARE AUTHORITY<br>
            <span style="font-size:14px;font-weight:500;">当前动作仅为软件模拟，不会控制真实机器人。</span>
            </div>
            """,
            visible=mock_mode,
            elem_id="operator-mock-safety-banner",
        )
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 指令 / 语音交互")
                user_chatbot = gr.Chatbot(
                    type="messages",
                    label="交互历史",
                    allow_tags=False,
                    value=[
                        {
                            "role": "assistant",
                            "content": "你好，可以输入指令或启动实时语音会话。",
                        }
                    ],
                    avatar_images=(
                        os.path.abspath("assets/icon/user.png"),
                        os.path.abspath("assets/icon/qwen.png"),
                    ),
                    height=500,
                )
                user_input = gr.MultimodalTextbox(
                    sources=["upload"],
                    file_types=["audio"],
                    placeholder="输入指令，例如：导航到前面的椅子",
                    label="指令输入",
                    interactive=True,
                    autofocus=True,
                    submit_btn="发送",
                    stop_btn=False,
                    elem_id="operator-command-input",
                )
                media_mode = "已启用" if chat_pipeline.asr is not None else "已关闭"
                gr.Markdown(
                    "实时语音请使用下方“开始语音会话”，无需录音后上传。"
                    "Codex 内置预览可能不提供物理麦克风，请使用系统 "
                    "Chrome/Edge 打开本页并授权；回形针仅保留用于调试音频文件。"
                    f" 当前文件上传 ASR：**{media_mode}**；实时 ASR 状态见下方。"
                )
                interaction_notice = gr.Markdown(
                    "最近交互：暂无",
                    elem_id="operator-interaction-notice",
                )
                with gr.Row():
                    voice_start_button = gr.Button(
                        "开始语音会话", elem_id="operator-voice-start"
                    )
                    voice_stop_button = gr.Button(
                        "结束语音会话", elem_id="operator-voice-stop"
                    )
                    voice_retry_button = gr.Button(
                        "重试语音连接", elem_id="operator-voice-retry"
                    )
                voice_status = gr.Markdown(
                    _voice_status_markdown(chat_pipeline.operator_snapshot()),
                    elem_id="operator-voice-status",
                )

            with gr.Column(scale=1):
                task_status = gr.Markdown(
                    "### Task 状态\n\n空闲",
                    elem_id="operator-task-status",
                )
                gr.Markdown("### 传感器预览")
                camera_status = gr.Markdown(
                    _camera_view_markdown(None, chat_pipeline.operator_snapshot()),
                    elem_id="operator-camera-status",
                )
                with gr.Tabs():
                    with gr.Tab("导航相机"):
                        nav_img_output = gr.Image(type="pil", height=280, visible=False)
                        nav_camera_placeholder = gr.Markdown(
                            "暂无画面：相机离线或尚未收到帧。",
                            visible=True,
                            elem_id="operator-nav-camera-placeholder",
                        )
                    with gr.Tab("机械臂 / 深度"):
                        manipulate_img_output = gr.Image(  # noqa: F841  # UI placeholder; kept for layout
                            type="pil", height=280, visible=False
                        )

        with gr.Row():
            with gr.Column(scale=2):
                task_timeline = gr.Markdown(
                    "### Task 时间线\n\n暂无任务事件。",
                    elem_id="operator-task-timeline",
                )
            with gr.Column(scale=1):
                telemetry_status = gr.Markdown(
                    "### 机器人与 Capability 状态\n\n等待遥测。",
                    elem_id="operator-telemetry-status",
                )

        stop_button = gr.Button(
            "停止当前任务",
            variant="stop",
            elem_id="operator-stop-button",
        )
        emergency_stop_button = gr.Button(
            "紧急停止机器人",
            variant="stop",
            elem_id="operator-emergency-stop",
        )
        camera_timer = gr.Timer(2.0, active=True)
        chat_timer = gr.Timer(1.0, active=True)
        refresh_timer = gr.Timer(5.0, active=True)

        user_input.submit(
            submit_operator_turn,
            inputs=[user_input, user_chatbot],
            outputs=[user_input, user_chatbot, interaction_notice],
            concurrency_limit=4,
            trigger_mode="multiple",
            api_name="submit_operator_turn",
        )
        stop_button.click(
            stop_operator_task,
            inputs=[],
            outputs=interaction_notice,
            concurrency_limit=4,
            api_name="stop_operator_task",
        )
        voice_start_button.click(
            start_voice_session,
            inputs=[],
            outputs=[voice_status, interaction_notice],
            concurrency_limit=1,
            api_name="start_voice_session",
            js=(
                "() => { window.ubrobotVoiceStart().catch(error => "
                "window.alert('麦克风或语音连接失败：' + error.message)); }"
            ),
        )
        voice_stop_button.click(
            stop_voice_session,
            inputs=[],
            outputs=[voice_status, interaction_notice],
            concurrency_limit=1,
            api_name="stop_voice_session",
            js="() => { window.ubrobotVoiceStop(); }",
        )
        voice_retry_button.click(
            start_voice_session,
            inputs=[],
            outputs=[voice_status, interaction_notice],
            concurrency_limit=1,
            api_name="retry_voice_session",
            js=(
                "() => { window.ubrobotVoiceRetry().catch(error => "
                "window.alert('语音重连失败：' + error.message)); }"
            ),
        )
        emergency_stop_button.click(
            emergency_stop_operator,
            inputs=[],
            outputs=[voice_status, interaction_notice],
            concurrency_limit=4,
            api_name="emergency_stop_operator",
            js="() => { window.ubrobotVoiceStop(); }",
        )
        camera_timer.tick(
            refresh_camera_once,
            inputs=[],
            outputs=[
                nav_img_output,
                nav_camera_placeholder,
                camera_status,
            ],
            concurrency_limit=1,
            api_name="camera_refresh",
            show_api=False,
        )
        chat_timer.tick(
            refresh_chat_once,
            inputs=[user_chatbot],
            outputs=[user_chatbot],
            concurrency_limit=1,
            api_name="chat_refresh",
            show_api=False,
        )
        refresh_timer.tick(
            refresh_console_once,
            inputs=[],
            outputs=[
                task_status,
                task_timeline,
                telemetry_status,
                voice_status,
            ],
            concurrency_limit=1,
            api_name="operator_refresh",
            show_api=False,
        )

    return demo.queue(default_concurrency_limit=4)


def create_fastapi():
    global chat_pipeline
    if chat_pipeline is None:
        media_enabled = (
            os.environ.get("UBROBOT_CHAT_MEDIA", "on").strip().lower() != "off"
        )
        chat_pipeline = ChatPipeline(initialize_media=media_enabled)
    pipeline = chat_pipeline

    @asynccontextmanager
    async def lifespan(_app):
        logger.info(
            "operator runtime ready backend=%s voice_provider=%s",
            pipeline.backend_name,
            pipeline.voice_runtime.snapshot().provider,
        )
        try:
            yield
        finally:
            await asyncio.to_thread(shutdown_pipeline, pipeline)

    app = FastAPI(lifespan=lifespan)
    shutdown_token = os.environ.get("UBROBOT_SHUTDOWN_TOKEN", "")

    @app.get("/api/health/live")
    async def health_live():
        return {"status": "live"}

    @app.get("/api/health/ready")
    async def health_ready():
        backend_name = pipeline.backend_name
        execution_mode = "mock" if backend_name == "cortex-mock" else "hardware-capable"
        return {
            "status": "ready",
            "backend": backend_name,
            "voice_provider": pipeline.voice_runtime.snapshot().provider,
            "execution_mode": execution_mode,
            "hardware_authority": bool(
                getattr(pipeline.backend, "hardware_authority", False)
            ),
            "capability_health": sanitized_capability_health(pipeline),
        }

    @app.get("/api/operator/snapshot")
    async def operator_snapshot():
        return {
            "type": "snapshot",
            "latest_event_id": pipeline.event_stream.latest_event_id(),
            "snapshot": pipeline.operator_snapshot(),
        }

    @app.get("/api/operator/capabilities")
    async def operator_capabilities():
        return {
            "hardware_authority": any(
                item["hardware_authority"]
                for item in pipeline.capability_registry.snapshot().values()
            ),
            "capabilities": pipeline.capability_registry.snapshot(),
        }

    @app.post("/api/operator/interactions")
    async def submit_operator_interaction(request: OperatorInteractionRequest):
        try:
            result = await asyncio.to_thread(
                execute_operator_interaction,
                request.text,
                source=request.source,
                correlation_id=request.correlation_id,
            )
        except Exception as exc:
            logger.exception(
                "operator API interaction failed source=%s correlation_id=%s",
                request.source,
                request.correlation_id or "generated",
            )
            raise HTTPException(
                status_code=409,
                detail={"error_type": type(exc).__name__, "message": str(exc)},
            ) from exc
        return _interaction_payload(result)

    @app.post("/api/operator/cancel")
    async def cancel_operator_task():
        active = pipeline.task_runtime.active_task()
        acknowledged = await asyncio.to_thread(pipeline.task_runtime.cancel_active)
        return {
            "acknowledged": acknowledged,
            "task_id": active.task_id if active is not None else None,
        }

    @app.post("/api/operator/emergency-stop")
    async def emergency_stop_api():
        acknowledged = await asyncio.to_thread(
            pipeline.voice_runtime.emergency_stop,
            "operator-api",
        )
        return {
            "acknowledged": acknowledged,
            "voice": pipeline.voice_runtime.snapshot().to_dict(),
        }

    @app.websocket("/api/operator/events")
    async def operator_events(websocket: WebSocket, after: int = 0):
        await websocket.accept()
        subscription = pipeline.event_stream.subscribe(
            after_event_id=max(0, after),
            queue_size=64,
        )
        logger.info("operator event stream opened after=%s", after)
        try:
            await websocket.send_json(
                {
                    "type": "snapshot",
                    "latest_event_id": pipeline.event_stream.latest_event_id(),
                    "replay_truncated": subscription.replay_truncated,
                    "snapshot": pipeline.operator_snapshot(),
                }
            )
            for event in subscription.replay:
                await websocket.send_json({"type": "event", "event": event.to_dict()})

            last_heartbeat = time.monotonic()
            while True:
                dropped = subscription.dropped_count(reset=True)
                if dropped:
                    await websocket.send_json(
                        {
                            "type": "gap",
                            "dropped": dropped,
                            "latest_event_id": pipeline.event_stream.latest_event_id(),
                            "snapshot": pipeline.operator_snapshot(),
                        }
                    )
                try:
                    event = await asyncio.to_thread(subscription.get, 0.5)
                except queue.Empty:
                    if time.monotonic() - last_heartbeat >= 5.0:
                        await websocket.send_json(
                            {
                                "type": "heartbeat",
                                "latest_event_id": pipeline.event_stream.latest_event_id(),
                            }
                        )
                        last_heartbeat = time.monotonic()
                    continue
                await websocket.send_json({"type": "event", "event": event.to_dict()})
                last_heartbeat = time.monotonic()
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            subscription.close()
            logger.info("operator event stream closed")

    @app.post("/api/admin/shutdown")
    async def request_shutdown(
        request: Request,
        x_ubrobot_shutdown_token: str = Header(default=""),
    ):
        if not shutdown_token:
            raise HTTPException(status_code=404, detail="shutdown control disabled")
        client_host = request.client.host if request.client is not None else ""
        if client_host not in {"127.0.0.1", "::1", "testclient"}:
            raise HTTPException(status_code=403, detail="local access required")
        if not secrets.compare_digest(x_ubrobot_shutdown_token, shutdown_token):
            raise HTTPException(status_code=403, detail="invalid shutdown token")
        server = getattr(app.state, "uvicorn_server", None)
        if server is None:
            raise HTTPException(status_code=503, detail="server control unavailable")
        server.should_exit = True
        logger.info("graceful shutdown requested by local launcher")
        return {"status": "stopping"}

    @app.websocket("/api/voice/stream")
    async def voice_stream(websocket: WebSocket):
        await websocket.accept()
        client = websocket.client
        logger.info("voice browser stream opened client=%s", client)
        loop = asyncio.get_running_loop()
        outgoing: asyncio.Queue[bytes | dict | None] = asyncio.Queue(maxsize=256)
        input_packets = 0
        accepted_packets = 0
        started = False

        def enqueue_output(item: bytes | dict) -> None:
            try:
                outgoing.put_nowait(item)
            except asyncio.QueueFull:
                logger.error("voice browser output queue overflow")

        def audio_sink(chunk: bytes) -> None:
            loop.call_soon_threadsafe(enqueue_output, chunk)

        def control_sink(control: str) -> None:
            loop.call_soon_threadsafe(enqueue_output, {"type": control})

        async def send_audio() -> None:
            try:
                while True:
                    chunk = await outgoing.get()
                    if chunk is None:
                        return
                    if isinstance(chunk, bytes):
                        await websocket.send_bytes(chunk)
                    else:
                        await websocket.send_json(chunk)
            except (WebSocketDisconnect, RuntimeError):
                logger.info("voice browser output stream closed")

        chat_pipeline.voice_runtime.set_audio_sink(audio_sink)
        chat_pipeline.voice_runtime.set_control_sink(control_sink)
        sender = asyncio.create_task(send_audio())
        try:
            await asyncio.to_thread(chat_pipeline.voice_runtime.start)
            started = True
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                audio = message.get("bytes")
                if audio:
                    input_packets += 1
                    accepted = chat_pipeline.voice_runtime.push_audio(audio)
                    accepted_packets += int(accepted)
                    if input_packets == 1:
                        logger.info(
                            "voice browser PCM received bytes=%s accepted=%s",
                            len(audio),
                            accepted,
                        )
                control = message.get("text")
                if control:
                    try:
                        command = json.loads(control)
                    except json.JSONDecodeError:
                        logger.warning("ignored invalid voice browser control")
                        continue
                    if command.get("type") == "playback.done":
                        chat_pipeline.voice_runtime.playback_finished()
                    elif command.get("type") == "microphone.level":
                        chat_pipeline.voice_runtime.update_microphone_level(
                            command.get("level", 0.0)
                        )
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.exception("voice browser stream failed")
            try:
                await websocket.close(code=1011, reason=str(exc)[:120])
            except RuntimeError:
                pass
        finally:
            chat_pipeline.voice_runtime.set_audio_sink(None)
            chat_pipeline.voice_runtime.set_control_sink(None)
            if started:
                await asyncio.to_thread(chat_pipeline.voice_runtime.stop)
            await outgoing.put(None)
            await sender
            logger.info(
                "voice browser stream closed packets=%s accepted=%s",
                input_packets,
                accepted_packets,
            )

    return gr.mount_gradio_app(app, create_gradio(), path="/")


if __name__ == "__main__":
    shutil.rmtree("./workspaces/results", ignore_errors=True)
    host = os.environ.get("UBROBOT_CHAT_HOST", "0.0.0.0")
    port = int(os.environ.get("UBROBOT_CHAT_PORT", "7863"))
    tls_enabled = os.environ.get("UBROBOT_CHAT_TLS", "on").strip().lower() != "off"
    try:
        require_port_available(host, port)
    except PortInUseError as exc:
        logger.error("%s", exc)
        raise SystemExit(2) from None
    logger.info(
        "starting operator console backend=%s media=%s url=%s://%s:%s",
        os.environ.get("UBROBOT_CHAT_BACKEND", "cortex"),
        os.environ.get("UBROBOT_CHAT_MEDIA", "on"),
        "https" if tls_enabled else "http",
        host,
        port,
    )
    application = create_fastapi()
    config = uvicorn.Config(
        application,
        host=host,
        port=port,
        log_level="info",
        ssl_keyfile="./assets/key.pem" if tls_enabled else None,
        ssl_certfile="./assets/cert.pem" if tls_enabled else None,
    )
    server = uvicorn.Server(config)
    application.state.uvicorn_server = server
    server.run()
