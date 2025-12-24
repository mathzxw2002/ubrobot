import os
import sys
import time
import random
import torch
import soundfile as sf
import dashscope
from dashscope.audio.tts_v2 import *
from dashscope.api_entities.dashscope_response import SpeechSynthesisResponse
#import edge_tts

#from src.GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
#from src.GPT_SoVITS.tools.i18n.i18n import I18nAuto, scan_language_list


'''@torch.no_grad()
class GPT_SoVits_TTS:
    def __init__(self, batch_size = 8):
        self.is_share = os.environ.get("is_share", "False")
        self.is_share = eval(self.is_share)

        if "_CUDA_VISIBLE_DEVICES" in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
        self.batch_size = batch_size
        self.is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
        self.gpt_path = os.environ.get("gpt_path", None)
        self.sovits_path = os.environ.get("sovits_path", None)
        self.cnhubert_base_path = os.environ.get("cnhubert_base_path", None)
        self.bert_path = os.environ.get("bert_path", None)
        self.version = os.environ.get("version", "v2")
        self.language = os.environ.get("language", "Auto")

        if self.language not in scan_language_list():
            self.language = "Auto"
        
        self.i18n = I18nAuto(language=self.language)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize TTS pipeline
        self.tts_config = TTS_Config("src/GPT_SoVITS/configs/tts_infer.yaml")
        self.tts_config.device = self.device
        self.tts_config.is_half = self.is_half
        self.tts_config.version = self.version

        if self.gpt_path is not None:
            self.tts_config.t2s_weights_path = self.gpt_path
        if self.sovits_path is not None:
            self.tts_config.vits_weights_path = self.sovits_path
        if self.cnhubert_base_path is not None:
            self.tts_config.cnhuhbert_base_path = self.cnhubert_base_path
        if self.bert_path is not None:
            self.tts_config.bert_base_path = self.bert_path
        
        print(self.tts_config)
        self.tts_pipeline = TTS(self.tts_config)

        self.dict_language = self.get_dict_language()
        self.cut_method = self.get_cut_method()

        # init and warm up
        self.init_infer()

    def get_dict_language(self):
        dict_language_v1 = {
            "中文": "all_zh",
            "英文": "en",
            "日文": "all_ja",
            "中英混合": "zh",
            "日英混合": "ja",
            "多语种混合": "auto",
        }

        dict_language_v2 = {
            "中文": "all_zh",
            "英文": "en",
            "日文": "all_ja",
            "粤语": "all_yue",
            "韩文": "all_ko",
            "中英混合": "zh",
            "日英混合": "ja",
            "粤英混合": "yue",
            "韩英混合": "ko",
            "多语种混合": "auto",
            "多语种混合(粤语)": "auto_yue",
        }

        return dict_language_v1 if self.version == 'v1' else dict_language_v2

    def get_cut_method(self):
        return {
            "不切": "cut0",
            "凑四句一切": "cut1",
            "凑50字一切": "cut2",
            "按中文句号切": "cut3",
            "按英文句号切": "cut4",
            "按标点符号切": "cut5",
        }

    def init_infer(self,
            ref_audio_path = 'data/audio/少女.wav', 
            prompt_text = "喜悦。哇塞！今天真是太棒了！悲伤。哎！生活怎么如此艰难。", 
            aux_ref_audio_paths=None, 
            text_lang="中英混合",
            prompt_lang="中文",
            top_k=5,
            top_p=1,
            temperature=1,
            text_split_method="按标点符号切",
            speed_factor=1.0,
            ref_text_free=False,
            split_bucket=True,
            fragment_interval=0.3,
            seed=-1,
            keep_random=True,
            return_fragment=False,
            parallel_infer=True,
            repetition_penalty=1.35):
            
        # Determine actual seed
        seed = -1 if keep_random else seed
        actual_seed = seed if seed not in [-1, "", None] else random.randrange(1 << 32)
        
        inputs = {
            "text_lang": self.dict_language[text_lang],
            "ref_audio_path": ref_audio_path,
            "aux_ref_audio_paths": [item.name for item in aux_ref_audio_paths] if aux_ref_audio_paths is not None else [],
            "prompt_text": prompt_text if not ref_text_free else "",
            "prompt_lang": self.dict_language[prompt_lang],
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "text_split_method": self.cut_method[text_split_method],
            "batch_size": self.batch_size,
            "speed_factor": float(speed_factor),
            "split_bucket": split_bucket,
            "return_fragment": return_fragment,
            "fragment_interval": fragment_interval,
            "seed": actual_seed,
            "parallel_infer": parallel_infer,
            "repetition_penalty": repetition_penalty,
        }
        self.tts_pipeline.init_run(inputs)
        for sampling_rate, audio_data in self.tts_pipeline.run(text = "首次infer，模型warm up。"):
            pass

    def infer(self, project_path, text, index = 0):  
        audio_path = f"{project_path}/audio"
        os.makedirs(audio_path, exist_ok=True)

        start_time = time.time()
        for sampling_rate, audio_data in self.tts_pipeline.run(text):
            output_wav_path = f"{audio_path}/llm_response_audio_{index}.wav"
            sf.write(output_wav_path, audio_data, sampling_rate)
            print(f"Save audio {output_wav_path}")
        print(f"Audio {index}:Cost {time.time()-start_time} secs")
        return output_wav_path
'''


class CosyVoice_API:
    def __init__(self):
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")  
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

'''class Edge_TTS:
    def __init__(self):
        self.voice = "en-GB-SoniaNeural" # use edge-tts --list-voices to see all available voices

    def infer(self, project_path, text, index = 0):
        try:
            audio_path = f"{project_path}/audio"
            os.makedirs(audio_path, exist_ok=True)
            output_wav_path = f"{audio_path}/llm_response_audio_{index}.wav"

            start_time = time.time()
            communicate = edge_tts.Communicate(text, self.voice)
            communicate.save(output_wav_path)
            print("[TTS] Edge TTS infer cost:", time.time()-start_time)
                
            return output_wav_path
        except Exception as e:
            print(f"[TTS] Edge TTS infer error: {e}")
            return None
'''
