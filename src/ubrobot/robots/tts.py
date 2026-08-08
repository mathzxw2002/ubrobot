import os
import time

import dashscope
from dashscope.audio.tts_v2 import *


class CosyVoice_API:
    def __init__(self):
        api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is required for CosyVoice_API")
        dashscope.api_key = api_key
        self.voice = "longwan"

    def infer(self, project_path, text, index = 0):
        try:
            audio_path = f"{project_path}/audio"
            os.makedirs(audio_path, exist_ok=True)
            output_wav_path = f"{audio_path}/llm_response_audio_{index}.wav"

            start_time = time.time()
            audio = SpeechSynthesizer(model="cosyvoice-v1", voice=self.voice).call(text)
            print("[TTS] API infer cost:", time.time()-start_time)
            with open(output_wav_path, 'wb') as f:
                f.write(audio)
            return output_wav_path
        except Exception as e:
            print(f"[TTS] API infer error: {e}")
            return None
