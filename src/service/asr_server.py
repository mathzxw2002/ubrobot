from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from funasr import AutoModel
import numpy as np
import asyncio
import soundfile as sf
import io

# 初始化FastAPI应用
app = FastAPI(title="FunASR 语音识别服务")

# 加载FunASR模型（全局初始化，避免重复加载）
# 轻量级模型：适合CPU，速度快；若有GPU，改为device="cuda:0"
'''model = AutoModel(
    model="iic/SenseVoiceSmall",
    vad_model="fsmn-vad",       # 语音活动检测
    punc_model="ct-punc",       # 标点恢复
    device="cpu",
    disable_update=True         # 禁用模型自动更新
)'''

model = AutoModel(
        model="paraformer-zh-streaming", 
        disable_update=True, 
        model_kwargs={"use_flash_attn":False}
)

# ---------------------- 前端页面（测试用） ----------------------
@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>FunASR 实时语音识别</title>
        <meta charset="utf-8">
    </head>
    <body>
        <h1>麦克风实时语音识别</h1>
        <button id="startBtn">开始录音</button>
        <button id="stopBtn" disabled>停止录音</button>
        <div id="result" style="margin-top:20px; font-size:18px;"></div>

        <script>
            // WebSocket连接
            let ws = null;
            // 音频录制相关
            let mediaRecorder = null;
            let audioChunks = [];

            // 开始录音
            document.getElementById('startBtn').addEventListener('click', async () => {
                // 初始化WebSocket
                ws = new WebSocket(`ws://${window.location.host}/ws/asr`);
                ws.onmessage = (event) => {
                    // 显示识别结果
                    document.getElementById('result').textContent = event.data;
                };
                ws.onclose = () => {
                    document.getElementById('result').textContent += "（连接已关闭）";
                };

                // 获取麦克风权限
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/wav' });
                
                // 实时发送音频数据
                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0 && ws.readyState === WebSocket.OPEN) {
                        // 将音频块转为ArrayBuffer发送
                        event.data.arrayBuffer().then(buffer => {
                            ws.send(buffer);
                        });
                    }
                };

                // 开始录制
                mediaRecorder.start(100);  // 每100ms发送一次数据
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
            });

            // 停止录音
            document.getElementById('stopBtn').addEventListener('click', () => {
                if (mediaRecorder) {
                    mediaRecorder.stop();
                    mediaRecorder = null;
                }
                if (ws) {
                    ws.close();
                    ws = null;
                }
                document.getElementById('startBtn').disabled = false;
                document.getElementById('stopBtn').disabled = true;
            });
        </script>
    </body>
    </html>
    """


# ---------------------- WebSocket流式识别接口 ----------------------
@app.websocket("/ws/asr")
async def websocket_asr(websocket: WebSocket):
    await websocket.accept()
    try:
        print("start processing audio")
        # 缓存音频数据，累计到一定长度再识别（避免频繁调用）
        audio_buffer = []
        chunk_size = 16000 * 0.5  # 0.5秒音频（16k采样率）
        sample_rate = 16000

        #chunk_size = [0, 10, 5] #[0, 10, 5] 600ms, [0, 8, 4] 480ms
        #encoder_chunk_look_back = 4 #number of chunks to lookback for encoder self-attention
        #decoder_chunk_look_back = 1 #number of encoder chunks to lookback for decoder cross-attention

        idx = 0
        while True:
            # 接收前端发送的音频二进制数据
            data = await websocket.receive_bytes()

            if not data:
                continue

            # 16bit PCM format
            audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) #/ 32768.0
            audio_buffer.extend(audio_np)

            # 累计足够数据后调用FunASR识别
            if len(audio_buffer) >= chunk_size:
                # 截取缓存的音频数据
                audio_chunk = np.array(audio_buffer[:int(chunk_size)])
                # 清空已处理的缓存（保留少量重叠，避免断句）
                audio_buffer = audio_buffer[int(chunk_size*0.8):]

                print("start asr...")
                # FunASR流式识别
                save_path = "audio_chunk_{}.wav".format(idx)  # 保存路径
                sf.write(
                    file=save_path,
                    data=audio_buffer,
                    samplerate=16000,
                    format="WAV",
                    subtype="PCM_16"
                )

                idx = idx + 1

                result = model.generate(
                    input=audio_chunk,
                    sample_rate=sample_rate,
                    is_final=False,  # 非最终帧（实时流式）
                    language="zh",   # 识别语言：zh/en/yue等
                    use_itn=True     # 数字/日期归一化
                )

                # 发送识别结果到前端
                print(result)
                if result and result[0]["text"]:
                    await websocket.send_text(result[0]["text"])

    except Exception as e:
        print(f"WebSocket错误: {e}")
        await websocket.send_text(f"识别出错：{str(e)}")
    finally:
        await websocket.close()

# ---------------------- HTTP非流式识别接口（文件上传） ----------------------
@app.post("/api/asr")
async def http_asr(file: UploadFile = File(...)):
    try:
        # 读取上传的音频文件
        audio_bytes = await file.read()
        # 调用FunASR识别（支持wav/mp3/flac等格式）
        result = model.generate(
            input=audio_bytes,
            audio_format=file.filename.split(".")[-1],  # 音频格式
            language="zh",
            use_itn=True
        )
        print("================== result...", result)
        return {"code": 200, "text": result[0]["text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"识别失败：{str(e)}")

# 启动服务
if __name__ == "__main__":
    import uvicorn
    # host=0.0.0.0 允许外网访问，port可自定义
    uvicorn.run("asr_server:app", host="0.0.0.0", port=8000, reload=True)
