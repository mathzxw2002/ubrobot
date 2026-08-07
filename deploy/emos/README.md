# EMOS container

This Compose file owns the persistent `emos` container while the EMOS CLI keeps
ownership of recipe execution. The container name must remain `emos` because the
CLI stops, starts, and executes recipes in a container with that name.

EMOS and external ROS 2 hardware drivers mount the same UDP-only Fast DDS profile.
This prevents Fast DDS from selecting shared memory across separate container IPC
namespaces, where discovery can succeed while topic data does not flow.

From the repository root on the Raspberry Pi:

```bash
docker compose -f deploy/emos/compose.yaml config
docker compose -f deploy/emos/compose.yaml build
docker compose -f deploy/emos/compose.yaml up -d
docker inspect emos --format '{{json .Config.Env}}'
docker inspect emos --format '{{json .Mounts}}'
```

The recipe data directory defaults to `/home/china/emos`. Set
`EMOS_DATA_DIR=/another/path` before invoking Compose when deploying as a
different host user. It is mounted at both `/emos` and `/home/china/emos`
because EMOS CLI 0.7.0 executes recipes from the first path but writes its log
files to the second.

The repository image extends the upstream Jazzy image with the native OMPL/FCL
runtime libraries, `kompass-core` 0.8.1, RealSense RGBD support,
`depthimage_to_laserscan`, and RTAB-Map RGB-D odometry required by
`vision_depth_follower`. It also bakes in:

- the kompass vision-follower setup-condition patch (upstream short-circuits
  the first action goal);
- the `emos_bringup` overlay with `vision_depth_bringup.launch.py` (RealSense,
  static TFs, RGB-D odometry, depth scan, `fix_detection_header` relay);
- `/usr/local/bin/emos-stack.sh`, the default container command, which starts
  the sensor chain and then the recipe, logging to
  `/home/china/emos/logs`. If any stack process dies, the container exits so
  `restart: always` recreates a clean stack.

The recipe executed by the supervisor is the one in the EMOS data dir
(`/emos/recipes/vision_depth_follower/recipe.py`, persisted on the host). The
repo keeps a reference copy at
`deploy/emos/recipes/vision_depth_follower/recipe.py` with the required
`detections_raw` wiring; the supervisor seeds it only when the data-dir recipe
is missing, it never overwrites an existing one.

For a manual debugging shell inside the running container:

```bash
docker exec -it emos bash
```

When launching a recipe by hand inside such a shell, continue to pass Fast DDS
explicitly:

```bash
emos run vision_depth_follower --rmw rmw_fastrtps_cpp --skip-sensor-check
```

Do not run `emos install` over this deployment. It creates the container with the
CLI's built-in Docker arguments and would discard the profile environment and
mount. Update the image with a backup followed by `docker compose pull` and
`docker compose up -d`, then repeat the cross-container topic test.

## Go2 dock deployment (CycloneDDS)

On the Go2 dock the emos stack MUST run `rmw_cyclonedds_cpp` so it can talk to
the `go2-piper-driver` hardware container (Go2's Unitree DDS transport is
CycloneDDS; see `deploy/go2-piper-driver/README`). Overlay the base compose with
the dock-specific CycloneDDS override:

```bash
docker compose -f deploy/emos/compose.yaml \
               -f deploy/emos/compose.dock-cyclonedds.yaml \
               -f deploy/emos/compose.cortex-navigation.yaml \
               up -d
```

The override sets `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` and mounts the same
`cyclonedds.xml` (eth0, spdp multicast) that `go2-piper-driver` uses, so both
containers join the same CycloneDDS domain on `ROS_DOMAIN_ID`. Without it the
emos `GraspObject` server cannot reach the piper driver's `/piper/joint_cmd`
topic (different RMW = different domain).

The Pi/LeKiwi stack keeps `rmw_fastrtps_cpp` (`deploy/emos/compose.yaml` alone)
and is unaffected.


## Cortex navigation mock validation

Use `test/cortex_navigation_mock_test.py` only with a disposable EMOS test
container running `cortex_navigation_bringup.launch.py start_sensors:=false`
and an independent LeKiwi container running `hardware_mode:=mock`. Neither
container may be privileged or map a device. The test embeds a deterministic
`/track_vision_target` fixture, so camera and VLM services are not required.

Available scenarios are:

```text
baseline goal cancel timeout orphan_client capability_loss raw_loss
stale_downstream driver_restart
```

Example inside the EMOS test container:

```bash
source /opt/ros/jazzy/setup.bash
source /opt/emos_overlay/setup.bash
python3 -u /tmp/cortex_navigation_mock_test.py \
  --scenario baseline --duration 30 --output /tmp/task8-baseline.json
python3 -u /tmp/cortex_navigation_mock_test.py \
  --scenario goal --output /tmp/task8-goal.json
```

Run each failure scenario against a freshly restarted EMOS test container.
`capability_loss` intentionally terminates `navigate_to_object_server` inside
that disposable container. For `driver_restart`, run the observer for at least
12 seconds and restart only the mock LeKiwi container after about 3 seconds.

The client always cancels owned Action goals in `finally` and waits for the
cancellation result. `orphan_client` is the deliberate exception at runtime:
the same client is SIGKILLed to prove the server timeout and lease expiry stop
motion even when cleanup cannot execute. Do not replace Action cancellation
with a shell process timeout.

The validated procedure and Raspberry Pi evidence are recorded in
`docs/validation/2026-07-30-cortex-navigation-mock.md`.

## Chat UI to Cortex no-motion smoke test

`test/chat_cortex_smoke_test.py` exercises the production Chat UI ROS Action
transport against disposable Cortex and bringup containers. Run the bringup
with `start_sensors:=false`, do not start a LeKiwi container, and require every
container to report an empty device list and `Privileged=false` before testing.

Copy the smoke client and `src/chat_ui/cortex_client.py` into a disposable UI
client container using the same ROS domain and UDP-only Fast DDS profile, then
run:

```bash
source /opt/ros/jazzy/setup.bash
source /opt/emos_overlay/setup.bash
python3 -u /tmp/chat_cortex_smoke_test.py \
  --output /tmp/task9-chat-cortex-smoke.json
```

The deterministic model fixture must return text only. The test fails if it
observes a non-empty `/navigation/command_lease`, a non-zero `/cmd_vel`, no
successful response, or no cancellation acknowledgement within two seconds.
An acknowledgement proves that the ROS Action server accepted cancellation;
it does not, by itself, prove the model HTTP request was interrupted.

The validated Raspberry Pi run and artifact locations are recorded in
`docs/validation/2026-07-30-chat-cortex-smoke.md`.


## Vision multimodal Q&A (describe_scene tool)

The recipe exposes a PLANNING-phase tool `vision_inspection.describe_scene`
that captures the latest RealSense color frame and asks a multimodal LLM
(Qwen-VL via DASHSCOPE) to describe the visible scene, enabling arbitrary
visual questions (e.g. "你能看到什么").

Configure via environment on the recipe container:

- `DASHSCOPE_API_KEY` — required; DASHSCOPE (Alibaba Model Studio) key
- `VISION_MODEL` — default `qwen-vl-max`
- `VISION_ENDPOINT` — default DASHSCOPE OpenAI-compatible endpoint
- `VISION_QUERY_PROMPT` — default Chinese scene-description prompt

The key must be injected at container start (e.g. from a 0600 file); it is
never baked into the image or written to the repository.
