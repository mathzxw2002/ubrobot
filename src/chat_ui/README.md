

# UBRobot Chat UI

The production request path submits ordinary text to the EMOS Cortex Action at
`/cortex_input_command`. It does not require a `nav:` prefix and does not
initialize `Go2Manager`, the local camera/VLM stack, or direct LeKiwi control.

The Operator Console routes typed text and ASR transcripts through an
independent `InteractionRuntime`. A single-active `TaskRuntime` owns Cortex
execution, cancellation, queued task metadata, parent/child links, and the
event timeline. `TelemetryHub` supplies timestamped sensor/capability state to
the UI without granting motion authority. See
`docs/adr/0004-operator-console-runtime-boundaries.md` for the architecture and
the local-to-distributed evolution path.

Start the primary Cortex path from the repository root:

```powershell
$env:UBROBOT_CHAT_BACKEND = "cortex"
python src/chat_ui/app.py
```

`cortex` is the default when `UBROBOT_CHAT_BACKEND` is unset. The UI and EMOS
must use the same ROS domain, RMW implementation, and Fast DDS profile.

## Robot Edge backend (M5 milestone)

The Robot Edge backend connects to a separate FastAPI service instead of
talking directly to ROS/EMOS Cortex. It adds authentication, replay protection,
lease management, and local safety supervision, while keeping ROS and hardware
SDKs on the robot side.

```powershell
$env:UBROBOT_CHAT_BACKEND = "robot-edge"
$env:UBROBOT_EDGE_URL = "http://127.0.0.1:8780"
$env:UBROBOT_EDGE_OPERATOR_ID = "operator"
$env:UBROBOT_EDGE_TOKEN = "test-token"  # or set UBROBOT_EDGE_TOKEN_FILE
python src/chat_ui/app.py
```

The Robot Edge service runs on the robot-side computer and exposes these
endpoints:

- `GET /v1/health/live`, `GET /v1/health/ready`
- `GET /v1/capabilities`
- `GET /v1/telemetry/snapshot`
- `GET /v1/events?after=<event_id>`
- `POST /v1/commands`
- `POST /v1/commands/{command_id}/cancel`
- `POST /v1/safety/stop`
- `POST /v1/lease/acquire`, `GET /v1/lease`

All command/control endpoints require a bearer token, a timestamp within the
request window, and a unique nonce to prevent replay attacks. Health endpoints
do not require authentication but reveal no secrets.

Run the Robot Edge service in fixture mode (no hardware):

```powershell
$env:UBROBOT_EDGE_MODE = "fixture"
$env:UBROBOT_EDGE_HARDWARE_AUTHORITY = "false"
python -m robot_edge.app
```

The fixture service accepts any token in test mode but rejects replayed nonces,
expired timestamps, and insufficient scopes in the real configuration.

For hardware deployment, set:

```powershell
$env:UBROBOT_EDGE_MODE = "hardware"
$env:UBROBOT_EDGE_HARDWARE_AUTHORITY = "true"
$env:UBROBOT_EDGE_SAFETY_CHECKLIST = "path/to/checklist"
```

Even with `HARDWARE_AUTHORITY=true`, M5 does not actually bind ROS or
hardware. Physical E-stop validation happens in later milestones.

## Offline development mode (Windows, no ROS)

Install the validated runtime and optional test dependencies from the
repository root:

```powershell
python -m pip install -r requirements-operator-console.txt
python -m pip install -r requirements-dev.txt
```

The Operator Console baseline is pinned to FastAPI 0.124.2, Gradio 5.50.0,
Starlette 0.47.2, Uvicorn 0.35.0, and websockets 15.0.1. The same versions are
declared in `requirements.in`, `requirements.txt`, and `pyproject.toml`.
`requirements-operator-console.txt` is the hardware-free Windows subset; the
full `requirements.txt` also contains Linux/ROS/ML and CUDA dependencies and
is not the workstation UI bootstrap file.

For UI development without a robot, ROS, or ASR/TTS models, run the in-process
mock backend with media disabled:

```powershell
$env:UBROBOT_CHAT_BACKEND = "cortex-mock"
$env:UBROBOT_CHAT_MEDIA = "off"
$env:UBROBOT_CHAT_TLS = "off"
$env:PYTHONPATH = "src;src/chat_ui"
python src/chat_ui/app.py
```

Open `http://127.0.0.1:7863`. The local HTTP switch avoids the self-signed
certificate warning during workstation-only testing; keep TLS enabled when
the console is exposed beyond localhost.

TLS uses per-machine self-signed dev credentials at `assets/key.pem` +
`assets/cert.pem`, which are **not tracked in git** (see `.gitignore`). On a
fresh clone, generate them once:

```bash
./scripts/generate_dev_cert.sh   # self-signed localhost cert (365 days)
```

Never commit these files; they are local development credentials.

Run the software-only test suite without ROS or hardware:

```powershell
python -m unittest discover -s tests/cortex_navigation -p "test_*.py" -v
```

`pytest` is available through `requirements-dev.txt`, but the suite remains
compatible with Python's built-in `unittest` runner.

### Managed Windows lifecycle

Use the launcher for normal workstation development. It writes a PID file and
separate stdout/stderr logs under `logs/`, rejects duplicate starts, checks the
readiness endpoint, and requests a token-protected graceful shutdown:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/operator_console.ps1 start
powershell -ExecutionPolicy Bypass -File scripts/operator_console.ps1 status
powershell -ExecutionPolicy Bypass -File scripts/operator_console.ps1 logs
powershell -ExecutionPolicy Bypass -File scripts/operator_console.ps1 stop
```

Use `-Port 7870` with every command to run an isolated development instance.
The launcher defaults unset variables to `cortex-mock`, media off, voice off,
and local HTTP. Set the environment explicitly before `start` to enable Qwen
or another supported mode.

Health endpoints:

- `GET /api/health/live` confirms that the HTTP process is alive.
- `GET /api/health/ready` reports the backend, execution mode, voice provider,
  and sanitized capability health. It never includes cloud credentials.

Structured operator controls used by the Gradio callbacks and automated
acceptance tests:

- `POST /api/operator/interactions` submits text through the same
  `InteractionRuntime -> TaskRuntime -> Cortex` entry point as the UI.
- `POST /api/operator/cancel` requests normal cancellation of the active task.
- `POST /api/operator/emergency-stop` bypasses task planning and calls the
  independent safety path.

The local shutdown endpoint is disabled for manually started processes. The
launcher enables it with a per-process token stored under `logs/`; requests
must come from loopback and include that token.

### Realtime operator state

Task, interaction, voice, and telemetry changes are published through one
bounded event stream. The browser loads a snapshot and then follows events by
monotonic cursor:

- `GET /api/operator/snapshot` returns the current serialized state and latest
  event ID.
- `WS /api/operator/events?after=<event_id>` replays retained events and then
  streams live updates.

Slow clients drop their oldest pending events and receive a fresh snapshot.
The Gradio timer now runs every five seconds only as a sensor/status fallback;
partial transcripts and runtime transitions normally update through the
WebSocket without waiting for that timer.

`cortex-mock` simulates Cortex feedback, multi-second navigation execution,
and bounded cancellation (Stop button), raising the same
`CortexRequestError("Plan aborted ...")` the real client produces on cancel.
`UBROBOT_CHAT_MEDIA=off` skips Fun_ASR/CosyVoice initialization, disables
audio-file transcription (marked inline), and closes the video queue after
the text reply. The Operator Console uses native Gradio input/chat components;
`modelscope-studio` is not part of its interaction path.

Useful offline checks after opening the console:

1. Submit `导航到前面的椅子` and observe `planning/running/succeeded` in the
   task status and timeline.
2. While the mock task is running, submit `任务进度怎么样？`; it is answered
   from TaskRuntime and does not create another Cortex request.
3. Submit `停一下` or use the stop button to exercise bounded cancellation.

The Raspberry Pi, Piper, Go2, and RealSense hardware remain out of scope for
this validation. Mock success is software evidence only.

Whenever `cortex-mock` is active, the console displays a persistent red
`MOCK / NO HARDWARE AUTHORITY` banner. Emergency stop events are marked
`critical`, bypass queued work, and supersede tasks already waiting in the
software queue. Cloud speech recognition remains a convenience input path;
it is not a hardware safety guarantee. An always-on local stop-word detector
and a physical E-stop are mandatory gates before hardware testing.

Run the complete M1-M4 software acceptance suite with one command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/validate_operator_console.ps1
```

The script runs the software suite plus a process-level test on a dynamically
allocated localhost port. It writes a timestamped, credential-free report to
`logs/validation/`. Navigation/grasp timing can be shortened in isolated Mock
tests with `UBROBOT_MOCK_NAV_DURATION_SEC` and
`UBROBOT_MOCK_REPLY_DELAY_SEC`; these variables never affect the real Cortex
backend.

## Hardware-ready adapter boundary (M4)

The workstation runtime exposes a serialized capability registry for
`navigation`, `grasp`, `observation`, `follow`, and `stop`:

- `GET /api/operator/capabilities` returns availability, health, execution
  mode, required resources, and hardware-authority state.
- Operator snapshots include the same descriptors and transport-neutral
  telemetry DTOs.
- Missing channels are `disconnected`, explicitly unavailable channels remain
  `unavailable`, and aged samples become `stale`.
- Mock and fixture capabilities can never claim hardware authority.
- TelemetryHub rejects binary frames and arbitrary SDK/runtime objects.

Only fixture adapters exist in this phase. They do not import or initialize
ROS, RealSense, Piper, or Go2 libraries. Future Robot Edge implementations
must stay behind these contracts; see
`docs/adr/0006-robot-edge-boundary.md` for API/ROS Action mappings,
authentication, navigation lease, disconnect, and emergency-stop rules.

## Qwen-Omni-Realtime half-duplex voice

The realtime voice path uses a provider-neutral `VoiceSessionManager`. Qwen is
the first adapter; a future Volcengine implementation can map streaming ASR
events to `interaction.request` and Cortex replies to streaming TTS without
changing `InteractionRuntime`, `TaskRuntime`, or robot capabilities.

Configure a Beijing Model Studio workspace and start the console with:

```powershell
$env:UBROBOT_CHAT_BACKEND = "cortex-mock"
$env:UBROBOT_CHAT_MEDIA = "off"
$env:UBROBOT_VOICE_PROVIDER = "qwen"
$env:DASHSCOPE_API_KEY = "<pay-as-you-go Model Studio API key>"
$env:DASHSCOPE_WORKSPACE_ID = "<workspace id>"
$env:UBROBOT_QWEN_REALTIME_REGION = "cn-beijing"
$env:UBROBOT_QWEN_REALTIME_PROXY = "direct"
$env:UBROBOT_CHAT_TLS = "off"
$env:PYTHONPATH = "src;src/chat_ui"
python src/chat_ui/app.py
```

Open `http://127.0.0.1:7863` in system Chrome or Edge and click **开始语音会话**
once to grant microphone permission. The browser streams 16 kHz PCM frames to
the local `/api/voice/stream` WebSocket; the API key remains server-side. Qwen
is allowed to call only `submit_interaction`, which routes through the existing
InteractionRuntime -> TaskRuntime -> Cortex path. Provider audio generated
before Cortex returns a tool result is discarded.

Normal turns are half-duplex: `listening -> thinking -> speaking -> listening`.
Audio frames received during `thinking` or `speaking` are ignored. The red UI
emergency-stop control remains available in every state, and spoken phrases
such as `紧急停止机器人` or `紧急叫停机器人` use TaskRuntime's safety path when
the normal listener is active. Always-on local keyword spotting and a physical
E-stop remain required before hardware validation; cloud speech recognition is
not a sufficient safety mechanism.

Qwen partial/final transcription and server VAD events are forwarded to the
operator event stream independently of tool-call completion. During reply
audio, the runtime remains in `speaking` until the browser confirms that all
scheduled PCM playback has completed; only then does it resume listening.
The browser reports microphone level at most five times per second.

If the voice provider or browser WebSocket disconnects, the browser retries
three times with 0.5/1/2 second backoff and then exposes the “重试语音连接”
button. Set the maximum billable cloud session duration when needed:

```powershell
$env:UBROBOT_QWEN_REALTIME_SESSION_TIMEOUT_SEC = "1800"
```

Provider credentials remain server-side. Operator snapshots, event payloads,
health responses, and logs never include the API key.

For cloud-free UI/state-machine validation use:

```powershell
$env:UBROBOT_VOICE_PROVIDER = "mock"
```

Never commit `DASHSCOPE_API_KEY`. The adapter reads credentials only from the
server environment.

`UBROBOT_QWEN_REALTIME_PROXY` defaults to `direct`. This avoids the
`websockets 15` proxy/TLS initialization failure seen with some Windows
WinINET localhost proxies. Set it to `auto` to use system proxy discovery, or
to an explicit proxy URL when the deployment requires one.

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
Use `scripts/generate_dev_cert.sh` (see the TLS note above) instead of a raw
openssl command. Equivalent one-off command:
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

