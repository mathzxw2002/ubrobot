import os
import torch
import time
import shutil
import threading
import queue
import gradio as gr

try:  # Package import for tests and `python -m chat_ui.app`.
    from .utils import get_timestamp_str, merge_audios, merge_frames_with_audio
    from .cortex_client import create_ros_cortex_client
except ImportError:  # Script compatibility: `python src/chat_ui/app.py`.
    from utils import get_timestamp_str, merge_audios, merge_frames_with_audio
    from cortex_client import create_ros_cortex_client


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

@torch.no_grad()
class ChatPipeline:
    def __init__(self, *, backend=None, initialize_media=True):
        if initialize_media:
            # Imported lazily so media-off dev mode needs no ASR/TTS deps.
            print("[1/3] Start initializing funasr")
            from ubrobot.robots.asr import Fun_ASR

            self.asr = Fun_ASR()

            print("[2/3] Start initializing tts")
            from ubrobot.robots.tts import CosyVoice_API

            self.tts_api = CosyVoice_API()
        else:
            self.asr = None
            self.tts_api = None
        
        self.timeout=180
        self.video_queue = queue.Queue()
        self.vlm_queue = queue.Queue()
        self.tts_queue = queue.Queue()
        self.cortex_feedback_queue = queue.Queue()
        self.chat_history = []
        self.stop = threading.Event()
        self._feedback_lock = threading.Lock()
        self.latest_cortex_feedback = ""

        if backend is not None:
            self.backend_name = "injected"
            self.backend = backend
        else:
            self.backend_name = os.environ.get(
                "UBROBOT_CHAT_BACKEND", "cortex"
            ).strip().lower()
            if self.backend_name == "cortex":
                print("[3/3] Start initializing Cortex client")
                self.backend = create_ros_cortex_client()
            elif self.backend_name == "cortex-mock":
                print("[3/3] Start initializing offline mock Cortex backend")
                try:
                    from .mock_backend import MockCortexBackend
                except ImportError:
                    from mock_backend import MockCortexBackend

                self.backend = MockCortexBackend()
            elif self.backend_name == "legacy":
                print("[3/3] Start initializing legacy Go2Manager")
                self.backend = _LegacyBackend()
            else:
                raise ValueError(
                    "UBROBOT_CHAT_BACKEND must be 'cortex', 'cortex-mock', "
                    "or 'legacy'"
                )
        print("[Done] Initialzation finished")
    
    def load_voice(self, avatar_voice = None, tts_module = None):
        start_time = time.time()
        avatar_voice = "longwan"
        
        yield gr.update(interactive=False, value=None)

        self.tts_api.voice = avatar_voice

        gr.Info("Avatar voice loaded.", duration = 2)
        yield gr.update(interactive=True, value=None)
        print(f"Load voice cost: {round(time.time()-start_time,2)}s")
    
    def flush_pipeline(self):
        print("Flushing pipeline....")
        self.video_queue = queue.Queue()
        self.vlm_queue = queue.Queue()
        self.tts_queue = queue.Queue()
        self.cortex_feedback_queue = queue.Queue()
        self.chat_history = []
        self.idx = 0
        self.start_time = None
        self.asr_cost = 0

    def stop_pipeline(self, user_processing_flag):
        if user_processing_flag:
            print("Stopping pipeline....")
            self.stop.set()
            self.backend.cancel_active()

            join_timeout = min(float(self.timeout), 5.0)
            for name in ("tts_thread", "ffmpeg_thread"):
                worker = getattr(self, name, None)
                if worker is not None:
                    worker.join(timeout=join_timeout)

            self.flush_pipeline()
            user_processing_flag = False

            self.stop.clear() 
            gr.Info("Stopping pipeline....", duration = 2)
            return user_processing_flag
        else:
            gr.Info("Pipeline is not running.", duration = 2)
            return user_processing_flag

    def _on_cortex_feedback(self, text):
        with self._feedback_lock:
            self.latest_cortex_feedback = text
        self.cortex_feedback_queue.put(text)

    def request_text(self, text):
        response = self.backend.execute(
            text,
            on_feedback=self._on_cortex_feedback,
        )
        if response:
            self.vlm_queue.put(response)
        return response

    def run_pipeline(self, user_input, user_messages):
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
        except Exception as e:
            print("make dir exception, ", {e})
        
        # Start pipeline
        gr.Info("Start processing.", duration = 2)
        try:
            # warm up
            media_on = self.tts_api is not None
            if media_on:
                self.tts_thread = threading.Thread(target=self.tts_worker, args=(self.project_path, tts_module, ))
                self.ffmpeg_thread = threading.Thread(target=self.ffmpeg_worker)
                self.tts_thread.start()
                self.ffmpeg_thread.start()

            # ASR
            user_input_txt = user_input.text
            if user_input.files:
                if self.asr is not None:
                    user_input_audio = user_input.files[0].path
                    user_input_txt += self.asr.infer(user_input_audio)
                else:
                    user_input_txt += " [ASR disabled: media off]"
            self.asr_cost = round(time.time()-self.start_time,2)

            print(f"[ASR] User input=========================================================: {user_input_txt}, cost: {self.asr_cost}s")
            
            user_messages.append({'role': 'user', 'content': user_input})
            print(user_messages)
            
            llm_response_txt = self.request_text(user_input_txt)
            
            if llm_response_txt:
                print(f"[LLM] Put into queue: {llm_response_txt}")

            if media_on:
                self.vlm_queue.put(None)
            else:
                # No TTS/ffmpeg workers in media-off dev mode; close the
                # video queue so yield_results finishes after feedback.
                self.video_queue.put(None)
            user_messages.append({'role': 'assistant', 'content': llm_response_txt})
            if len(user_messages) > 10:
                user_messages.pop(0)

            if media_on:
                self.tts_thread.join()
                self.ffmpeg_thread.join()

            # Remove frames
            if self.stop.is_set():
                print("Stop pipeline......")
            else:
                print("Finish pipeline......")
            return user_messages

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            gr.Error(f"An error occurred: {str(e)}")
            return None

    def yield_results(self, user_input, user_chatbot, user_processing_flag):
        user_processing_flag = True
        user_chatbot.append([
            {
                "text": user_input.text,
                "files": user_input.files,
            },
            {
                "text": "开始生成......\n",
            }
        ])
        yield gr.update(interactive=False, value=None), user_chatbot, user_processing_flag

        time.sleep(1)
        index = 0
        videos_dir_path = None
        start_time = time.time()
        print("[Listener] Start yielding results from queue.")

        try:
            while not self.stop.is_set():
                try:

                    #if index >= len(self.chat_history):
                    #    break
                    video_result = self.video_queue.get(timeout=1)

                    #llm_response_audio = self.tts_queue.get(timeout=1)

                    if not video_result:
                    #if not llm_response_audio:
                        break
                    videos_dir_path = os.path.dirname(video_result.video_path)
                    user_chatbot[-1][1]["text"]+=self.chat_history[index]

                    yield gr.update(interactive=False, value=None), user_chatbot, user_processing_flag
                    gr.Info(f"Streaming video_{index} from queue.", duration = 1)
                    print(f"[Listener] Streaming video_{index} from queue.")
                    time.sleep(2)
                    index += 1
                    start_time = time.time()
                    
                except queue.Empty: 
                    try:
                        status = self.cortex_feedback_queue.get_nowait()
                        user_chatbot[-1][1]["text"] = status + "\n"
                        yield (
                            gr.update(interactive=False, value=None),
                            user_chatbot,
                            user_processing_flag,
                        )
                    except queue.Empty:
                        pass
                    if time.time() - start_time > self.timeout:
                        gr.Info("Timeout, stop listening video stream queue.")
                        break

                except Exception as e:
                    gr.Error(f"An error occurred: {str(e)}")

            # Merge all videos
            if not self.stop.is_set() and videos_dir_path:
                merged_audio_path = merge_audios(videos_dir_path)
                llm_response_txt = user_chatbot[-1][1]["text"] + f"""<audio src="{merged_audio_path}" autoplay></audio>\n"""
                user_chatbot[-1][1] = {
                        "text": llm_response_txt,
                        "flushing": False
                    }

            if self.stop.is_set():
                user_chatbot[-1][1]["text"]+="\n停止生成，请稍等......"
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            gr.Error(f"An error occurred: {str(e)}")

        finally:
            yield gr.update(interactive=True, value=None), user_chatbot, user_processing_flag

            if videos_dir_path: 
                results_path = os.path.dirname(videos_dir_path)
                print(f"Remove results: {results_path}")
                shutil.rmtree(results_path, ignore_errors=True)
            user_processing_flag = False

    def tts_worker(self, project_path, tts_module):
        start_time = time.time()
        index = 0
        tts_module = "CosyVoice"
        
        while not self.stop.is_set():
            print("waiting vlm response...")
            try:
                llm_response_txt = self.vlm_queue.get(timeout=180)
                self.chat_history.append(llm_response_txt)
                print(f"[TTS] Get chunk from llm_queue: {llm_response_txt}, llm_queue size: {self.vlm_queue.qsize()}, chat_history {self.chat_history} ")
                if not llm_response_txt:
                    break
                infer_start_time = time.time()

                llm_response_audio = self.tts_api.infer(project_path=project_path, text=llm_response_txt, index = index)
                self.tts_queue.put(llm_response_audio)
                print(f"----------------[TTS] tts_queue size:{self.tts_queue.qsize()}")
                start_time = time.time()
                index+=1
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
                infer_start_time = time.time()
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
        return observer()
