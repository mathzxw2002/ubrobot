import os
import re
import subprocess
import cv2
import time
from pathlib import Path
from datetime import datetime
import wave
from dashscope.audio.tts_v2 import *

from dataclasses import dataclass

@dataclass
class VideoInfo:
    video_path: str  # 视频路径（若仅保留音频，此处可改为音频路径）
    audio_path: str  # 音频路径


def merge_frames_with_audio(audio_path, fps = 25):
    video_idx = audio_path.split("/")[-1].split("_")[-1].split(".")[0]
    print(f"[Real-time Inference] Merging frames with audio on {video_idx}")

    video_path = str(Path(audio_path).parent.parent / "videos" / f"{video_idx}.ts")
    #frame_path = str(Path(audio_path).parent.parent / "frames" / f"{video_idx}")
    start_time = time.time()
    
    '''ffmpeg_command = [
        'ffmpeg',
        '-framerate', str(fps),
        '-i', f"{frame_path}/%08d.jpg",
        '-i', audio_path,
        '-c:v', 'libx264',     
        '-shortest',
        '-f', 'mpegts',    
        '-y',     
        video_path
    ]'''

    # just keep audio
    ffmpeg_command = [
        'ffmpeg',
        '-i', audio_path,
        '-c:a', 'copy',
        '-vn',
        '-y',
        video_path
    ]

    print("+++++++++++++++++++ video_path:", video_path, audio_path)

    #subprocess.run(ffmpeg_command, check=True)
    print(f"[Real-time Inference] Merging frames with audio costs {time.time()-start_time}s")
    #return video_path
    
    return VideoInfo(video_path, audio_path)

def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    return round(duration, 2)

def split_into_sentences(text, sentence_split_option):
    text = ''.join(text.splitlines())
    sentence_endings = re.compile(r'[。！？.!?]')
    sentences = sentence_endings.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]
    split_count = int(sentence_split_option)
    return ['。'.join(sentences[i:i+split_count]) for i in range(0, len(sentences), split_count)]

def get_timestamp_str():
    fmt = "%Y%m%d_%H%M%S"
    current_time = datetime.now()
    folder_name = current_time.strftime(fmt)
    return folder_name

def merge_videos(video_folder_path, suffix = '.mp4'):
    output_path = os.path.join(video_folder_path, f'merged_video{suffix}')
    file_list_path = os.path.join(video_folder_path, 'video_list.txt')

    print("-------------------- merge_videos...", output_path, file_list_path)

    audio_folder_path = str(Path(video_folder_path).parent / "audio" )

    def extract_index(filename):
        index = filename.split('.')[0].split('_')[-1]
        return int(index) 

    with open(file_list_path, 'w') as file_list:
        ts_files = [f for f in os.listdir(video_folder_path) if f.endswith('.ts')]
        ts_files.sort(key=extract_index)

        for filename in ts_files:
            file_list.write(f"file '{filename}'\n")

    ffmpeg_command = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', file_list_path,
        '-c', 'copy',
        '-c:v', 'libx264',
        '-c:a', 'aac', 
        '-y',
        output_path
    ]

    subprocess.run(ffmpeg_command, check=True)


    output_audio_path = os.path.join(video_folder_path, f'merged_audio{suffix}')
    audio_file_list_path = os.path.join(audio_folder_path, 'audio_list.txt')
    with open(audio_file_list_path, 'w') as audio_file_list:
        wav_files = [f for f in os.listdir(audio_folder_path) if f.endswith('.wav')]
        wav_files.sort(key=extract_index)
        for filename in wav_files:
            audio_file_list.write(f"file '{filename}'\n")

    print("+++++++++++++++++++++++++, ", audio_file_list_path, output_audio_path)

    ffmpeg_command = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', file_list_path,
        '-c', 'copy',
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-y',
        output_path
    ]
    
    ffmpeg_command = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', audio_file_list_path,
        '-c:a', 'copy',
        '-y',
        output_audio_path
    ]
    subprocess.run(ffmpeg_command, check=True)

    return output_path

def merge_audios(video_folder_path, suffix = '.wav'):
    audio_folder_path = str(Path(video_folder_path).parent / "audio" )

    def extract_index(filename):
        index = filename.split('.')[0].split('_')[-1]
        return int(index)

    output_audio_path = os.path.join(video_folder_path, f'merged_audio{suffix}')
    audio_file_list_path = os.path.join(audio_folder_path, 'audio_list.txt')
    with open(audio_file_list_path, 'w') as audio_file_list:
        wav_files = [f for f in os.listdir(audio_folder_path) if f.endswith('.wav')]
        wav_files.sort(key=extract_index)
        for filename in wav_files:
            audio_file_list.write(f"file '{filename}'\n")

    print("+++++++++++++++++++++++++, ", audio_file_list_path, output_audio_path)

    ffmpeg_command = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', audio_file_list_path,
        '-c:a', 'copy',
        '-y',
        output_audio_path
    ]
    subprocess.run(ffmpeg_command, check=True)

    return output_audio_path
