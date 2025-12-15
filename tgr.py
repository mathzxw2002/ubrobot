import gradio as gr
import numpy as np
import os
import shutil
from fastrtc import WebRTC, ReplyOnPause
from fastrtc.pause_detection.silero import get_silero_model
from huggingface_hub import snapshot_download, hf_hub_download

from fastrtc.pause_detection.silero import SileroVADModel


LOCAL_VAD_MODEL_FILE = "/home/china/silero_vad.onnx"
#vad_model = SileroVADModel(model_path=LOCAL_VAD_MODEL_PATH)

# ========== 关键配置 ==========
# 1. 本地Silero VAD模型路径（你下载的onnx文件）
# 2. HuggingFace默认缓存路径（让fastrtc从这里读取）
HF_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")
# 3. 模型在HuggingFace的路径（固定值）
HF_MODEL_REPO = "freddyaboulton/silero-vad"
HF_MODEL_FILENAME = "silero_vad.onnx"

# ========== 手动将本地模型复制到HF缓存目录 ==========
def setup_local_vad_model():
    """将本地模型文件复制到fastrtc默认读取的缓存路径"""
    # 构建缓存路径
    cache_model_dir = os.path.join(
        HF_CACHE_DIR,
        f"models--{HF_MODEL_REPO.replace('/', '--')}",
        "blobs"
    )
    cache_model_path = os.path.join(cache_model_dir, HF_MODEL_FILENAME)

    # 创建缓存目录
    os.makedirs(cache_model_dir, exist_ok=True)

    # 复制本地模型到缓存路径
    if not os.path.exists(cache_model_path):
        shutil.copy2(LOCAL_VAD_MODEL_FILE, cache_model_path)
        print(f"本地模型已复制到缓存：{cache_model_path}")
    else:
        print(f"模型已存在于缓存：{cache_model_path}")

    # 验证文件
    if not os.path.exists(cache_model_path):
        raise FileNotFoundError(f"模型复制失败：{cache_model_path}")
    return cache_model_path

# ========== 初始化配置 ==========
# 1. 先配置本地模型缓存
setup_local_vad_model()

def response(audio: tuple[int, np.ndarray]):
    """This function must yield audio frames"""
    ...
    yield audio


with gr.Blocks() as demo:
    gr.HTML(
    """
    <h1 style='text-align: center'>
    Chat (Powered by WebRTC ⚡️)
    </h1>
    """
    )
    with gr.Column():
        with gr.Group():
            audio = WebRTC(
                mode="send-receive",
                modality="audio",
            )
        audio.stream(fn=ReplyOnPause(response),
                    inputs=[audio], outputs=[audio],
                    time_limit=60)
demo.launch()
