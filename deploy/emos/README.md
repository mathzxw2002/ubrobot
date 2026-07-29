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
