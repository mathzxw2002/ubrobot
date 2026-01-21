import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import torch
import time
import numpy as np
import copy
import shutil
import threading
import queue
import time
import gradio as gr
import ffmpeg
import subprocess
import threading
import gradio as gr
import pandas as pd

from utils import get_timestamp_str, merge_videos, merge_audios, merge_frames_with_audio
from ubrobot.robots.tts import CosyVoice_API
from ubrobot.robots.asr import Fun_ASR
#from ubrobot.robots.llm import Qwen_API
from ubrobot.robots.ubrobot import Go2Manager
from ubrobot.robots.arm_action import PoseTransformer
from ubrobot.robots.vlm import RobotVLM

@torch.no_grad()
class ChatPipeline:
    def __init__(self):

        print(f"[2/4] Start initializing funasr")
        self.asr = Fun_ASR()

        print(f"[3/4] Start initializing qwen")
        #self.llm = Qwen_API()
        self.llm = RobotVLM()

        print(f"[4/4] Start initializing tts")
        #self.tts = GPT_SoVits_TTS()
        self.tts_api = CosyVoice_API()
        
        print("[Done] Initialzation finished")
        self.timeout=180
        self.video_queue = queue.Queue()
        self.llm_queue = queue.Queue()
        self.tts_queue = queue.Queue()
        self.chat_history = []
        self.stop = threading.Event()
        
        self.robot_arm = PoseTransformer()
        #self.manager = None

    
    def load_voice(self, avatar_voice = None, tts_module = None):#, ref_audio_path = None):
        start_time = time.time()
        avatar_voice = avatar_voice.split(" ")[0]
        yield gr.update(interactive=False, value=None)
        # GPT-SoVits
        '''if tts_module == "GPT-SoVits":
            ref_audio_path = f'data/audio/{avatar_voice}.wav'
            prompt_text = self.asr.infer(ref_audio_path)
            self.tts.init_infer(ref_audio_path, prompt_text)
        # CosyVoice
        else:'''
        self.tts_api.voice = avatar_voice  

        gr.Info("Avatar voice loaded.", duration = 2)
        yield gr.update(interactive=True, value=None)
        print(f"Load voice cost: {round(time.time()-start_time,2)}s")
    
    def flush_pipeline(self):
        print("Flushing pipeline....")
        self.video_queue = queue.Queue()
        self.llm_queue = queue.Queue()
        self.tts_queue = queue.Queue()
        self.chat_history = []
        self.idx = 0
        self.start_time = None
        self.asr_cost = 0

    def stop_pipeline(self, user_processing_flag):
        if user_processing_flag:
            print("Stopping pipeline....")
            self.stop.set()
            time.sleep(1)

            self.tts_thread.join()
            self.ffmpeg_thread.join()

            self.flush_pipeline()
            user_processing_flag = False

            self.stop.clear() 
            gr.Info("Stopping pipeline....", duration = 2)
            return user_processing_flag
        else:
            gr.Info("Pipeline is not running.", duration = 2)
            return user_processing_flag

    def run_pipeline(self, user_input, user_messages, chunk_size, avatar_name, tts_module, chat_mode):
        self.flush_pipeline()
        self.start_time = time.time()
        avatar_name = avatar_name.split(" ")[0]
        self.project_path = f"./workspaces/results/{avatar_name}/{get_timestamp_str()}"
        #project_path = f"./workspaces/results/{get_timestamp_str()}"
        user_input_audio = None
        #tts_module = 'GPT-SoVits' if user_input_audio else tts_module

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
            self.tts_thread = threading.Thread(target=self.tts_worker, args=(self.project_path, tts_module, ))
            self.ffmpeg_thread = threading.Thread(target=self.ffmpeg_worker)
            self.tts_thread.start()
            self.ffmpeg_thread.start()

            # ASR
            user_input_txt = user_input.text
            if user_input.files:
                user_input_audio = user_input.files[0].path
                user_input_txt += self.asr.infer(user_input_audio)
            self.asr_cost = round(time.time()-self.start_time,2)

            print(f"[ASR] User input=========================================================: {user_input_txt}, cost: {self.asr_cost}s")

            # LLM streaming out
            #llm_response_txt, user_messages = self.llm.infer_stream(user_input_txt, user_messages, self.llm_queue, chunk_size, chat_mode)

            manipulate_img_output = self.robot_arm.get_observation()
            llm_response_txt, user_messages = self.llm.infer_cosmos_reason(user_input_txt, user_messages, self.llm_queue, manipulate_img_output)

            print("============================================llm_response_txt", llm_response_txt)

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
                    
                    #videos_dir_path = str(Path(llm_response_audio).parent.parent / "videos" )
                    #videos_dir_path = project_path + "/videos"

                    user_chatbot[-1][1]["text"]+=self.chat_history[index]

                    yield gr.update(interactive=False, value=None), user_chatbot, user_processing_flag
                    gr.Info(f"Streaming video_{index} from queue.", duration = 1)
                    print(f"[Listener] Streaming video_{index} from queue.")
                    time.sleep(2)
                    index += 1
                    start_time = time.time()
                    
                except queue.Empty: 
                    if time.time() - start_time > self.timeout:
                        gr.Info("Timeout, stop listening video stream queue.")
                        break

                except Exception as e:
                    gr.Error(f"++++++++++++++++++++++++++++++An error occurred: {str(e)}")

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
            print(f"===================tttttttttttttttttttttttttttt, An error occurred: {str(e)}")
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
        while not self.stop.is_set():
            print("waiting vlm response...")
            try:
                llm_response_txt = self.llm_queue.get(timeout=180)
                self.chat_history.append(llm_response_txt)
                print(f"-----------------[TTS] Get chunk from llm_queue: {llm_response_txt}, llm_queue size: {self.llm_queue.qsize()}, chat_history {self.chat_history} ")
                if not llm_response_txt:
                    break
                infer_start_time = time.time()

                #if tts_module == "GPT-SoVits":
                #    llm_response_audio = self.tts.infer(project_path=project_path, text=llm_response_txt, index = index)
                #else:
                llm_response_audio = self.tts_api.infer(project_path=project_path, text=llm_response_txt, index = index)  
                
                self.tts_queue.put(llm_response_audio)

                print("---------------------------------self.tts_queue.put(llm_response_audio), ", llm_response_audio)

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


    def get_robot_arm_image_observation(self):
        return self.robot_arm.get_observation()
    
    #def get_front_image_observation(self):
    #    return self.



# 实例化         
#chat_pipeline = ChatPipeline()
