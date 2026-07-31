

# UBRobot Chat UI

The production request path submits ordinary text to the EMOS Cortex Action at
`/cortex_input_command`. It does not require a `nav:` prefix and does not
initialize `Go2Manager`, the local camera/VLM stack, or direct LeKiwi control.

Start the primary Cortex path from the repository root:

```powershell
$env:UBROBOT_CHAT_BACKEND = "cortex"
python src/chat_ui/app.py
```

`cortex` is the default when `UBROBOT_CHAT_BACKEND` is unset. The UI and EMOS
must use the same ROS domain, RMW implementation, and Fast DDS profile.

## Offline development mode (Windows, no ROS)

For UI development without a robot, ROS, or ASR/TTS models, run the in-process
mock backend with media disabled:

```powershell
$env:UBROBOT_CHAT_BACKEND = "cortex-mock"
$env:UBROBOT_CHAT_MEDIA = "off"
$env:PYTHONPATH = "src;src/chat_ui"
python src/chat_ui/app.py
```

`cortex-mock` simulates Cortex feedback, multi-second navigation execution,
and bounded cancellation (Stop button), raising the same
`CortexRequestError("Plan aborted ...")` the real client produces on cancel.
`UBROBOT_CHAT_MEDIA=off` skips Fun_ASR/CosyVoice initialization, disables
audio-file transcription (marked inline), and closes the video queue after
the text reply. `modelscope-studio` must match `requirements.txt` (1.6.1);
2.x renames `Chatbot` to `ProChatbot` and breaks the UI.

## Legacy rollback mode

The previous keyword-routing implementation remains available only as an
explicit rollback/research path:

```powershell
$env:UBROBOT_CHAT_BACKEND = "legacy"
python src/chat_ui/app.py
```

Legacy mode constructs `Go2Manager`, reconnects local camera, VLM, and robot
dependencies, and retains the historical `nav:`/`grasp:` behavior. Do not use
it as the production Cortex path or enable it during controlled hardware
validation without repeating the legacy safety preflight.

## Upstream and historical environment notes

https://github.com/mathzxw2002/VideoChat

forked from: https://github.com/Henry-23/VideoChat

## Bug Fix

### 1, TypeError: Client.__init__() got an unexpected keyword argument 'proxies'
pip install httpx==0.27.2
https://blog.csdn.net/weixin_44003104/article/details/144375184


###
  File "/home/sany/.local/lib/python3.12/site-packages/gradio_client/utils.py", line 880, in get_type
    if "const" in schema:
       ^^^^^^^^^^^^^^^^^
TypeError: argument of type 'bool' is not iterable

#pip install gradio==6.2.0 gradio_client==2.0.2


gradio-5.4.0 gradio-client-1.4.2



###

pip install numpy==1.26.4 scipy==1.11.4 librosa==0.10.1 opencv-python==4.8.1.78  --break-system-packages


pip install huggingface-hub==0.25.1 --break-system-packages


tokenizers 0.19.1 requires huggingface-hub<1.0,>=0.16.4, but you have huggingface-hub 1.2.3 which is incompatible.
transformers 4.44.1 requires huggingface-hub<1.0,>=0.23.2, but you have huggingface-hub 1.2.3 which is incompatible.


## gradio启动 报错 TypeError: argument of type ‘bool‘ is not iterable
https://blog.csdn.net/qq_63234089/article/details/146914002



# Generate Encrypt
penssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

