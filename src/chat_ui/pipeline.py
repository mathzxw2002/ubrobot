import logging
import os
import queue
import shutil
import threading
import time

import gradio as gr

logger = logging.getLogger("ubrobot.operator_console.pipeline")

try:
    import torch

    def no_grad():
        return torch.no_grad()
except ModuleNotFoundError:
    # The hardware-free Operator Console does not run local torch inference.
    def no_grad():
        return lambda function: function


try:  # Package import for tests and `python -m chat_ui.app`.
    from .adapters.telemetry import (
        CapabilityHealthTelemetry,
        FixtureTelemetryAdapter,
        TelemetryState,
    )
    from .capability_registry import ExecutionMode, create_default_registry
    from .cortex_client import create_ros_cortex_client
    from .event_stream import EventStream
    from .interaction_runtime import InteractionRuntime
    from .task_runtime import TaskRuntime
    from .telemetry import TelemetryHub
    from .utils import get_timestamp_str, merge_audios, merge_frames_with_audio
    from .voice_runtime import (
        DisabledVoiceProvider,
        MockVoiceProvider,
        VoiceSessionManager,
    )
except ImportError:  # Script compatibility: `python src/chat_ui/app.py`.
    from adapters.telemetry import (
        CapabilityHealthTelemetry,
        FixtureTelemetryAdapter,
        TelemetryState,
    )
    from capability_registry import ExecutionMode, create_default_registry
    from cortex_client import create_ros_cortex_client
    from event_stream import EventStream
    from interaction_runtime import InteractionRuntime
    from task_runtime import TaskRuntime
    from telemetry import TelemetryHub
    from voice_runtime import (
        DisabledVoiceProvider,
        MockVoiceProvider,
        VoiceSessionManager,
    )

    from utils import get_timestamp_str, merge_audios, merge_frames_with_audio


class _LegacyBackend:
    """Explicit rollback adapter; never constructed by the primary path."""

    def __init__(self):
        from ubrobot.robots.ubrobot import Go2Manager

        self.manager = Go2Manager()
        self.manager.start_threads()

    def execute(self, task, *, on_feedback):
        on_feedback("legacy backend")
        return self.manager.agent_response(task)

    def cancel_active(self):
        self.manager.nav_by_user_instruction("stop")
        return True

    def get_robot_observation(self):
        return self.manager.visualize_robot_observation()


class ChatPipeline:
    def __init__(self, *, backend=None, initialize_media=True, voice_provider=None):
        if initialize_media:
            # Imported lazily so media-off dev mode needs no ASR/TTS deps.
            logger.info("[1/3] Start initializing funasr")
            from ubrobot.robots.asr import Fun_ASR

            self.asr = Fun_ASR()

            logger.info("[2/3] Start initializing tts")
            from ubrobot.robots.tts import CosyVoice_API

            self.tts_api = CosyVoice_API()
        else:
            self.asr = None
            self.tts_api = None

        self.timeout = 180
        self.video_queue = queue.Queue()
        self.vlm_queue = queue.Queue()
        self.tts_queue = queue.Queue()
        self.cortex_feedback_queue = queue.Queue()
        self.chat_history = []
        self.stop = threading.Event()
        self._feedback_lock = threading.Lock()
        self._completed_replies: list[tuple[str, str]] = []
        self._completed_lock = threading.Lock()
        self.latest_cortex_feedback = ""

        if backend is not None:
            self.backend_name = "injected"
            self.backend = backend
        else:
            self.backend_name = (
                os.environ.get("UBROBOT_CHAT_BACKEND", "cortex").strip().lower()
            )
            if self.backend_name == "cortex":
                logger.info("[3/3] Start initializing Cortex client")
                self.backend = create_ros_cortex_client()
            elif self.backend_name == "cortex-mock":
                logger.info("[3/3] Start initializing offline mock Cortex backend")
                try:
                    from .mock_backend import MockCortexBackend
                except ImportError:
                    from mock_backend import MockCortexBackend

                self.backend = MockCortexBackend(
                    nav_duration_sec=float(
                        os.environ.get("UBROBOT_MOCK_NAV_DURATION_SEC", "4.0")
                    ),
                    reply_delay_sec=float(
                        os.environ.get("UBROBOT_MOCK_REPLY_DELAY_SEC", "0.3")
                    ),
                )
            elif self.backend_name == "robot-edge":
                logger.info("[3/3] Start initializing Robot Edge backend")
                try:
                    from .adapters.robot_edge import RobotEdgeBackend
                except ImportError:
                    from adapters.robot_edge import RobotEdgeBackend

                self.backend = RobotEdgeBackend(
                    edge_url=os.environ.get(
                        "UBROBOT_EDGE_URL", "http://127.0.0.1:8780"
                    ),
                    operator_id=os.environ.get("UBROBOT_EDGE_OPERATOR_ID", "operator"),
                    token_file=os.environ.get("UBROBOT_EDGE_TOKEN_FILE"),
                    token=os.environ.get("UBROBOT_EDGE_TOKEN"),
                )
            elif self.backend_name == "legacy":
                logger.info("[3/3] Start initializing legacy Go2Manager")
                self.backend = _LegacyBackend()
            else:
                raise ValueError(
                    "UBROBOT_CHAT_BACKEND must be 'cortex', 'cortex-mock', "
                    "'robot-edge', or 'legacy'"
                )
        self.event_stream = EventStream()
        execution_modes = {
            "cortex-mock": ExecutionMode.MOCK,
            "injected": ExecutionMode.FIXTURE,
            "cortex": ExecutionMode.REMOTE,
            "robot-edge": ExecutionMode.REMOTE,
            "legacy": ExecutionMode.HARDWARE,
        }
        simulated_capabilities = (
            ("navigation", "grasp", "follow", "stop")
            if self.backend_name == "cortex-mock"
            else ()
        )
        self.capability_registry = create_default_registry(
            execution_mode=execution_modes[self.backend_name],
            simulated_capabilities=simulated_capabilities,
        )
        self.task_runtime = TaskRuntime(
            self.backend,
            event_publisher=self.event_stream.publish,
        )
        self.interaction_runtime = InteractionRuntime(
            self.task_runtime,
            event_publisher=self.event_stream.publish,
        )
        self.telemetry_hub = TelemetryHub(
            event_publisher=self.event_stream.publish,
        )
        # Robot Edge telemetry and capability clients. The backend constructor
        # already raises if no token is configured, so reaching here means a
        # valid operator token is available; reuse the same loader for clients.
        self.edge_telemetry_client = None
        self.edge_capability_client = None
        if self.backend_name == "robot-edge":
            try:
                from .adapters.robot_edge import RobotEdgeBackend as _EdgeBackend
            except ImportError:
                from adapters.robot_edge import RobotEdgeBackend as _EdgeBackend
            try:
                from .adapters.robot_edge_telemetry import (
                    RobotEdgeCapabilityClient,
                    RobotEdgeTelemetryClient,
                )
            except ImportError:
                from adapters.robot_edge_telemetry import (
                    RobotEdgeCapabilityClient,
                    RobotEdgeTelemetryClient,
                )
            edge_url = os.environ.get("UBROBOT_EDGE_URL", "http://127.0.0.1:8780")
            # No default token: _load_token returns "" when nothing is
            # configured, and the telemetry clients reject an empty token.
            edge_token = _EdgeBackend._load_token(
                os.environ.get("UBROBOT_EDGE_TOKEN_FILE"),
                os.environ.get("UBROBOT_EDGE_TOKEN"),
            )
            local_hardware_permitted = (
                os.environ.get("UBROBOT_EDGE_LOCAL_HARDWARE_PERMITTED", "false").lower()
                == "true"
            )
            self.edge_telemetry_client = RobotEdgeTelemetryClient(
                edge_url=edge_url,
                token=edge_token,
                telemetry_hub=self.telemetry_hub,
            )
            self.edge_capability_client = RobotEdgeCapabilityClient(
                edge_url=edge_url,
                token=edge_token,
                capability_registry=self.capability_registry,
                local_hardware_permitted=local_hardware_permitted,
            )
            self.edge_telemetry_client.start()
            self.edge_capability_client.start()
        else:
            # Use fixture telemetry for other backends
            self.telemetry_adapter = FixtureTelemetryAdapter(
                {
                    "capability_health": CapabilityHealthTelemetry(
                        state=TelemetryState.AVAILABLE,
                        source="capability_registry",
                        capabilities=self.capability_registry.snapshot(),
                        detail="serialized operator capability inventory",
                    )
                }
            )
            self.telemetry_adapter.publish_all(self.telemetry_hub)
        self.voice_provider = voice_provider or self._create_voice_provider()
        self.voice_runtime = VoiceSessionManager(
            self.voice_provider,
            interaction_handler=lambda text: self.request_text(text, source="voice"),
            contextual_interaction_handler=lambda text,
            correlation_id: self.request_text(
                text,
                source="voice",
                correlation_id=correlation_id,
            ),
            emergency_stop_handler=lambda source: self.task_runtime.emergency_stop(
                source=source
            ),
            event_publisher=self.event_stream.publish,
        )
        logger.info("[Done] Initialization finished")

    @staticmethod
    def _create_voice_provider():
        provider_name = os.environ.get("UBROBOT_VOICE_PROVIDER", "off").strip().lower()
        if provider_name in {"", "off", "disabled"}:
            return DisabledVoiceProvider()
        if provider_name == "mock":
            return MockVoiceProvider()
        if provider_name == "qwen":
            try:
                from .qwen_realtime import QwenOmniRealtimeProvider, QwenRealtimeConfig
            except ImportError:
                from qwen_realtime import QwenOmniRealtimeProvider, QwenRealtimeConfig
            return QwenOmniRealtimeProvider(QwenRealtimeConfig.from_env())
        raise ValueError("UBROBOT_VOICE_PROVIDER must be 'off', 'mock', or 'qwen'")

    def load_voice(self, avatar_voice=None, tts_module=None):
        start_time = time.time()
        avatar_voice = "longwan"

        yield gr.update(interactive=False, value=None)

        self.tts_api.voice = avatar_voice

        gr.Info("Avatar voice loaded.", duration=2)
        yield gr.update(interactive=True, value=None)
        logger.info("Load voice cost: %.2fs", round(time.time() - start_time, 2))

    def flush_pipeline(self):
        logger.info("Flushing pipeline...")
        self.video_queue = queue.Queue()
        self.vlm_queue = queue.Queue()
        self.tts_queue = queue.Queue()
        self.cortex_feedback_queue = queue.Queue()
        self.chat_history = []
        self.idx = 0
        self.start_time = None
        self.asr_cost = 0

    def stop_pipeline(self, user_processing_flag):
        if user_processing_flag or self.task_runtime.active_task() is not None:
            logger.info("Stopping pipeline...")
            self.stop.set()
            self.task_runtime.cancel_active()

            join_timeout = min(float(self.timeout), 5.0)
            for name in ("tts_thread", "ffmpeg_thread"):
                worker = getattr(self, name, None)
                if worker is not None:
                    worker.join(timeout=join_timeout)

            self.flush_pipeline()
            user_processing_flag = False

            self.stop.clear()
            gr.Info("Stopping pipeline....", duration=2)
            return user_processing_flag
        else:
            gr.Info("Pipeline is not running.", duration=2)
            return user_processing_flag

    def _on_cortex_feedback(self, text):
        with self._feedback_lock:
            self.latest_cortex_feedback = text
        self.cortex_feedback_queue.put(text)

    def record_completed(self, text: str, reply: str) -> None:
        """Record a background interaction result for the UI timer to drain."""
        with self._completed_lock:
            self._completed_replies.append((text, reply))

    def take_completed(self) -> list[tuple[str, str]]:
        """Return and clear completed background replies (thread-safe)."""
        with self._completed_lock:
            items = list(self._completed_replies)
            self._completed_replies.clear()
            return items

    def request_interaction(self, text, *, source="text", correlation_id=None):
        """Run the transport-neutral interaction path and return its result."""
        result = self.interaction_runtime.handle(
            text,
            source=source,
            correlation_id=correlation_id,
            on_feedback=self._on_cortex_feedback,
        )
        response = result.reply
        if response:
            self.vlm_queue.put(response)
        return result

    def request_text(self, text, *, source="text", correlation_id=None):
        """Compatibility wrapper returning only the interaction reply."""
        result = self.request_interaction(
            text,
            source=source,
            correlation_id=correlation_id,
        )
        response = result.reply
        return response

    @no_grad()
    def run_pipeline(self, user_input, user_messages):
        # A status/cancel turn may arrive while the main task is active. Do not
        # destroy its queues or feedback timeline in that case.
        if self.task_runtime.active_task() is None:
            self.flush_pipeline()
        self.start_time = time.time()
        avatar_name = "Avatar1"
        self.project_path = f"./workspaces/results/{avatar_name}/{get_timestamp_str()}"
        user_input_audio = None
        tts_module = "CosyVoice"

        try:
            os.makedirs(self.project_path, exist_ok=True)
            videos_path = f"{self.project_path}/videos"
            os.makedirs(videos_path, exist_ok=True)
        except Exception:
            logger.exception("make dir failed")

        # Start pipeline
        gr.Info("Start processing.", duration=2)
        try:
            # warm up
            media_on = self.tts_api is not None
            owns_media_workers = media_on and not any(
                getattr(getattr(self, name, None), "is_alive", lambda: False)()
                for name in ("tts_thread", "ffmpeg_thread")
            )
            if owns_media_workers:
                self.tts_thread = threading.Thread(
                    target=self.tts_worker,
                    args=(
                        self.project_path,
                        tts_module,
                    ),
                )
                self.ffmpeg_thread = threading.Thread(target=self.ffmpeg_worker)
                self.tts_thread.start()
                self.ffmpeg_thread.start()

            # ASR
            user_input_txt = user_input.text
            interaction_source = "text"
            if user_input.files:
                interaction_source = "voice"
                if self.asr is not None:
                    user_input_audio = user_input.files[0].path
                    user_input_txt += self.asr.infer(user_input_audio)
                else:
                    user_input_txt += " [ASR disabled: media off]"
            self.asr_cost = round(time.time() - self.start_time, 2)

            logger.info("[ASR] user input: %s, cost: %.2fs", user_input_txt, self.asr_cost)
            user_messages.append({"role": "user", "content": user_input})
            logger.debug("user messages: %s", user_messages)

            llm_response_txt = self.request_text(
                user_input_txt,
                source=interaction_source,
            )

            if llm_response_txt:
                logger.info("[LLM] Put into queue: %s", llm_response_txt)

            if owns_media_workers:
                self.vlm_queue.put(None)
            elif not media_on:
                # No TTS/ffmpeg workers in media-off dev mode; close the
                # video queue so yield_results finishes after feedback.
                self.video_queue.put(None)
            user_messages.append({"role": "assistant", "content": llm_response_txt})
            if len(user_messages) > 10:
                user_messages.pop(0)

            if owns_media_workers:
                self.tts_thread.join()
                self.ffmpeg_thread.join()

            # Remove frames
            if self.stop.is_set():
                logger.info("Stop pipeline")
            else:
                logger.info("Finish pipeline")
            return user_messages

        except Exception as e:
            logger.exception("pipeline error")
            gr.Error(f"An error occurred: {str(e)}")
            return None

    def yield_results(self, user_input, user_chatbot, user_processing_flag):
        user_processing_flag = True
        user_chatbot.append(
            [
                {
                    "text": user_input.text,
                    "files": user_input.files,
                },
                {
                    "text": "开始生成......\n",
                },
            ]
        )
        # Keep the interaction channel available for status/cancel utterances.
        yield (
            gr.update(interactive=True, value=None),
            user_chatbot,
            user_processing_flag,
        )

        time.sleep(1)
        index = 0
        videos_dir_path = None
        start_time = time.time()
        logger.debug("[Listener] Start yielding results from queue.")

        try:
            while not self.stop.is_set():
                try:
                    # if index >= len(self.chat_history):
                    #    break
                    video_result = self.video_queue.get(timeout=1)

                    # llm_response_audio = self.tts_queue.get(timeout=1)

                    if not video_result:
                        # if not llm_response_audio:
                        break
                    videos_dir_path = os.path.dirname(video_result.video_path)
                    user_chatbot[-1][1]["text"] += self.chat_history[index]

                    yield (
                        gr.update(interactive=False, value=None),
                        user_chatbot,
                        user_processing_flag,
                    )
                    gr.Info(f"Streaming video_{index} from queue.", duration=1)
                    logger.debug("[Listener] Streaming video_%s from queue.", index)
                    time.sleep(2)
                    index += 1
                    start_time = time.time()

                except queue.Empty:
                    try:
                        status = self.cortex_feedback_queue.get_nowait()
                        user_chatbot[-1][1]["text"] = status + "\n"
                        yield (
                            gr.update(interactive=True, value=None),
                            user_chatbot,
                            user_processing_flag,
                        )
                    except queue.Empty:
                        pass
                    if time.time() - start_time > self.timeout:
                        gr.Info("Timeout, stop listening video stream queue.")
                        break

                except Exception as e:
                    logger.exception("listener video-stream error")
                    gr.Error(f"An error occurred: {str(e)}")

            # Merge all videos
            if not self.stop.is_set() and videos_dir_path:
                merged_audio_path = merge_audios(videos_dir_path)
                llm_response_txt = (
                    user_chatbot[-1][1]["text"]
                    + f"""<audio src="{merged_audio_path}" autoplay></audio>\n"""
                )
                user_chatbot[-1][1] = {"text": llm_response_txt, "flushing": False}

            if self.stop.is_set():
                user_chatbot[-1][1]["text"] += "\n停止生成，请稍等......"
        except Exception as e:
            logger.exception("listener error")
            gr.Error(f"An error occurred: {str(e)}")

        finally:
            yield (
                gr.update(interactive=True, value=None),
                user_chatbot,
                user_processing_flag,
            )

            if videos_dir_path:
                results_path = os.path.dirname(videos_dir_path)
                logger.info("Remove results: %s", results_path)
                shutil.rmtree(results_path, ignore_errors=True)
            user_processing_flag = False

    def tts_worker(self, project_path, tts_module):
        start_time = time.time()
        index = 0

        while not self.stop.is_set():
            logger.debug("tts_worker waiting for vlm response...")
            try:
                llm_response_txt = self.vlm_queue.get(timeout=180)
                self.chat_history.append(llm_response_txt)
                logger.debug(
                    "[TTS] chunk from llm_queue: %s, queue size: %s, history: %s",
                    llm_response_txt,
                    self.vlm_queue.qsize(),
                    self.chat_history,
                )
                if not llm_response_txt:
                    break
                llm_response_audio = self.tts_api.infer(
                    project_path=project_path, text=llm_response_txt, index=index
                )
                self.tts_queue.put(llm_response_audio)
                logger.debug("[TTS] tts_queue size: %s", self.tts_queue.qsize())
                start_time = time.time()
                index += 1
            except queue.Empty:
                if time.time() - start_time > self.timeout:
                    gr.Info("TTS Timeout")
                    break
        self.tts_queue.put(None)

    def ffmpeg_worker(self):
        start_time = time.time()
        while not self.stop.is_set():
            try:
                llm_response_audio = self.tts_queue.get(timeout=1)
                if not llm_response_audio:
                    break
                video_result = merge_frames_with_audio(llm_response_audio)
                self.video_queue.put(video_result)
                start_time = time.time()
            except queue.Empty:
                if time.time() - start_time > self.timeout:
                    gr.Info("ffmpeg Timeout")
                    break
        self.video_queue.put(None)

    def get_robot_observation(self):
        observer = getattr(self.backend, "get_robot_observation", None)
        if observer is None:
            return None, None
        navigation_image, manipulation_image = observer()
        self.telemetry_hub.publish(
            "camera",
            self._observation_metadata(navigation_image),
        )
        self.telemetry_hub.publish(
            "depth",
            self._observation_metadata(manipulation_image),
        )
        return navigation_image, manipulation_image

    def operator_snapshot(self):
        """Return transport-neutral state for Gradio or a future remote UI."""
        return {
            "tasks": self.task_runtime.snapshot(),
            "interactions": [
                turn.to_dict() for turn in self.interaction_runtime.turns()
            ],
            "telemetry": self.telemetry_hub.snapshot(),
            "capabilities": self.capability_registry.snapshot(),
            "voice": self.voice_runtime.snapshot().to_dict(),
        }

    @staticmethod
    def _observation_metadata(value):
        if value is None:
            return {"available": False}
        size = getattr(value, "size", None)
        shape = getattr(value, "shape", None)
        width = height = None
        if isinstance(size, (tuple, list)) and len(size) >= 2:
            width, height = int(size[0]), int(size[1])
        return {
            "available": True,
            "size": list(size) if isinstance(size, tuple) else size,
            "shape": list(shape) if shape is not None else None,
            # The console's own refresh overwrites the Robot Edge camera_info
            # payload; keep width/height so the telemetry row keeps showing
            # the resolution instead of "-".
            "width": width,
            "height": height,
        }
