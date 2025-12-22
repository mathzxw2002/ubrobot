import gradio as gr
from fastrtc import WebRTC, ReplyOnPause
import traceback # Import traceback
import numpy as np

import os
# 1. 清空所有代理环境变量（优先级最高）
for key in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY"]:
    os.environ.pop(key, None)
# 2. 强制本地地址直连（覆盖系统代理规则）
os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0,192.168.0.0/16,::1"
# 3. 禁用Gradio在线资源加载（避免触发代理请求）
#os.environ["GRADIO_OFFLINE_MODE"] = "1"
#os.environ["HF_HUB_OFFLINE"] = "1"

def response(audio: tuple[int, np.ndarray]):
    """This function must yield audio frames"""
    try:
        # Add your actual processing logic here
        print("Processing audio frame...")
        if audio is None:
            print("Received None audio input.")
        else:
            sample_rate, data = audio
            print(f"Sample rate: {sample_rate}, Data shape: {data.shape}")

        print("===============================================")
        yield audio # Yield the input back for testing
    except Exception as e:
        print(f"An ERROR occurred in response function: {e}")
        traceback.print_exc() # Print the full traceback for debugging
        # You might need to yield something to keep the generator alive, 
        # though the error will likely still disrupt the stream.
        yield audio 

class LocalPauseDetector:
    def __init__(self, func, pause_threshold=0.5, silence_threshold=0.01):
        self.func = func
        self.pause_threshold = pause_threshold  # 暂停检测阈值（秒）
        self.silence_threshold = silence_threshold  # 静音音量阈值

    def __call__(self, audio):
        if audio is None:
            return None
        sample_rate, data = audio
        # 计算音频音量（判断是否静音）
        audio_energy = np.max(np.abs(data))
        if audio_energy < self.silence_threshold:
            # 检测到静音，触发处理函数
            return self.func(audio)
        # 有声音时直接返回原音频（实时回声）
        return audio

# ========== 3. 音频处理函数（流式返回） ==========
def audio_processor(audio: tuple[int, np.ndarray]):
    """
    流式处理音频，支持实时回声/自定义处理
    :param audio: (采样率, 音频数据np数组)
    :return: 处理后的音频元组
    """
    try:
        if audio is None:
            return None
        sample_rate, data = audio
        # 可选：添加自定义处理（如降噪、变声等）
        processed_data = data.astype(np.float32)  # 强制float32兼容Gradio
        return (sample_rate, processed_data)
    except Exception as e:
        print(f"音频处理错误：{e}")
        traceback.print_exc()
        return audio
    
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
                #autoplay=True,
                rtc_configuration={
                    "iceServers": [
                        # 国内可用的STUN服务器
                        {"urls": "stun:stun.qq.com:3478"},
                        {"urls": "stun:stun.cloudflare.com:3478"}
                    ]
                }
            )
        audio.stream(fn=ReplyOnPause(response),
                    inputs=[audio], 
                    outputs=[audio],
                    time_limit=60,
                    concurrency_limit=1,
        )

        print("RTC配置：", audio.rtc_configuration) 

        audio_component = gr.Audio(
            sources="microphone",  # 麦克风输入
            type="numpy",         # 输出numpy数组（方便处理）
            streaming=True,       # 流式处理（实时交互核心）
            autoplay=True,        # 自动播放返回的音频（回声效果）
            label="麦克风/扬声器",
            elem_id="audio-stream",
            show_label=False,
            show_download_button=False,
            show_share_button=False,
        )

        # 绑定流式处理（模拟 WebRTC 的 send-receive 模式）
        audio_component.stream(
            fn=LocalPauseDetector(audio_processor),  # 绑定暂停检测
            inputs=[audio_component],
            outputs=[audio_component],
            time_limit=60,  # 单次流最大时长（秒）
            concurrency_limit=1  # 单线程处理（避免冲突）
        )


demo.launch(
    server_name="0.0.0.0",
    server_port=7867,
    share=False,
    inbrowser=True,
    show_error=True,
    #static_files={
    #    "/manifest.json": "./manifest.json"  # 映射manifest.json路径
    #},
    #ssl_certfile="./ub_cert.pem",
    #ssl_keyfile="./ub_key.pem",
    #ssl_verify=False
)