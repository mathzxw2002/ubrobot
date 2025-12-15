import json
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import soundfile
import os
import logging
from fastapi.middleware.cors import CORSMiddleware
from tempfile import NamedTemporaryFile
from funasr import AutoModel
from starlette.responses import StreamingResponse

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FunASR")

# 模型参数（与你原始代码一致）
chunk_size = [0, 10, 5]  # [0, 10, 5] 对应600ms分块
encoder_chunk_look_back = 4
decoder_chunk_look_back = 1

# 初始化模型（仅加载一次）
# 修改为本地模型路径
model_path = "./models/paraformer-zh-streaming"

logger.info(f"🧠 加载本地 FunASR 模型: {model_path} ...")
model = AutoModel(model=model_path)

app = FastAPI()

# 允许跨域（前端访问需要）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 实际生产环境建议限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 支持的音频格式（根据soundfile支持的格式扩展）
ALLOWED_EXTENSIONS = {".wav", ".flac", ".ogg"}


def allowed_file(filename: str) -> bool:
    """检查文件扩展名是否合法"""
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS

@app.post("/transcribe")
async def asr_stream_endpoint(file: UploadFile = File(...)):
    """接收音频文件并流式返回每个 chunk 的识别结果"""

    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="不支持的文件格式")

    with NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        temp_file.write(await file.read())
        temp_path = temp_file.name
        print("temp_path{}", temp_path)
    try:
        speech, sample_rate = soundfile.read(temp_path)
        chunk_stride = chunk_size[1] * 960
        cache = {}
        total_chunk_num = int((len(speech) - 1) / chunk_stride + 1)

        # 使用 FastAPI 的 StreamingResponse
        async def generate():
            for i in range(total_chunk_num):
                start = i * chunk_stride
                end = (i + 1) * chunk_stride
                speech_chunk = speech[start:end]
                is_final = (i == total_chunk_num - 1)

                res = model.generate(
                    input=speech_chunk,
                    cache=cache,
                    is_final=is_final,
                    chunk_size=chunk_size,
                    encoder_chunk_look_back=encoder_chunk_look_back,
                    decoder_chunk_look_back=decoder_chunk_look_back
                )
                # 提取 text 字段内容
                if res and isinstance(res, list):
                    texts = [item.get("text", "") for item in res]
                    full_text = " ".join(texts).strip()  # 或者用 ''.join 拼接成完整句子
                else:
                    full_text = ""
                print("full_text{}", full_text)
                yield json.dumps({"text": full_text}, ensure_ascii=False) + "\n"

        return StreamingResponse(generate(), media_type="application/json")

    except Exception as e:
        logger.error(f"识别失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务端错误: {str(e)}")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
