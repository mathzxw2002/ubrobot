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
`vision_depth_follower`.

Continue to pass Fast DDS explicitly when launching a recipe:

```bash
emos run vision_depth_follower --rmw rmw_fastrtps_cpp --skip-sensor-check
```

Do not run `emos install` over this deployment. It creates the container with the
CLI's built-in Docker arguments and would discard the profile environment and
mount. Update the image with a backup followed by `docker compose pull` and
`docker compose up -d`, then repeat the cross-container topic test.
