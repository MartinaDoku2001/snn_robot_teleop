# snn_robot_teleop — Phase 0: two simulated Rover Minis

Two [Rover Robotics Mini](https://github.com/RoverRobotics/roverrobotics_ros2)
robots in Gazebo, namespaced `/robot1` and `/robot2`, each with its own keyboard
teleop. Everything runs in Docker, so every Linux distribution gets the same
environment. Phase 0 is infrastructure only: it has no controllers and no
networking or transmission logic.

| | |
|---|---|
| ROS | 2 Humble (Ubuntu 22.04 inside the container) |
| Simulator | **Gazebo Fortress** (Ignition Gazebo 6) via `ros_gz`. This is what the vendor `roverrobotics_gazebo` package depends on (`ros_gz_sim` and `ros_gz_bridge`), not Gazebo Classic. |
| Base image | `osrf/ros:humble-desktop-full`, pinned by digest in the [Dockerfile](Dockerfile). It already ships Fortress 6.18 and ros_gz 0.244. |
| Robot drive | The vendor Gazebo **DiffDrive system plugin**, not ros2_control |

## Prerequisites

- Linux with a graphical X11 session. Wayland desktops work through XWayland (GNOME and KDE provide it by default).
- Docker Engine with the Compose v2 plugin (`docker compose version`). Your user must be able to run `docker`.
- `xhost`. Package names: Ubuntu/Debian `x11-xserver-utils`, Arch `xorg-xhost`, Fedora `xhost`.
- Optional, NVIDIA only: [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Quick start

```bash
git clone <this repo> && cd snn_robot_teleop
./scripts/xhost_setup.sh        # one-time per login session (see below)
docker compose up -d            # builds the image the first time (~5-10 min)
docker compose logs -f sim      # first start runs colcon build; wait for "Summary: 5 packages finished"
./scripts/sim.sh                # Gazebo + RViz with robot1 and robot2
```

In two more terminals, start one teleop session per robot:

```bash
./scripts/teleop.sh robot1      # or ./scripts/teleop_robot1.sh
./scripts/teleop.sh robot2      # or ./scripts/teleop_robot2.sh
```

Each teleop is a `teleop_twist_keyboard` publishing to `/<robot>/cmd_vel`. The
terminal must have keyboard focus: `i` drives forward, `,` backward, `j`/`l`
turn, `k` stops, and `q`/`z` change speed.

`./scripts/up.sh` does the xhost step and `docker compose up -d` together. It
also picks the GPU mode automatically; see below.

Stop with `Ctrl-C` in the sim terminal, then run `docker compose down`. Build
artifacts persist in Docker volumes. `docker compose down -v` also wipes them,
and the next start rebuilds.

### The xhost step (why and what)

The container runs GUI apps as root and connects to your X server through
`/tmp/.X11-unix`. X denies that connection by default.
`./scripts/xhost_setup.sh` runs `xhost +local:root`, which allows local root
clients only (not the network). You need it once per login session. Revoke it
with `xhost -local:root`.

## Launch commands (inside the container)

`docker compose exec sim bash` opens a shell with ROS and the workspace
already sourced.

```bash
ros2 launch rover_multi_bringup multi_mini.launch.py              # 2 robots, Gazebo GUI + RViz
ros2 launch rover_multi_bringup multi_mini.launch.py rviz:=false  # no RViz
ros2 launch rover_multi_bringup multi_mini.launch.py gui:=false rviz:=false   # headless
ros2 launch rover_multi_bringup single_mini.launch.py             # one namespaced robot
ros2 launch roverrobotics_gazebo mini_gazebo.launch.py            # the unmodified vendor launch (1 robot, global names)
```

`multi_mini.launch.py` arguments:

| arg | default | meaning |
|---|---|---|
| `gui` | `true` | Gazebo GUI client. `false` runs the server headless. |
| `rviz` | `true` | Start RViz with a two-robot config. |
| `world` | `arena.sdf` | A file in `rover_multi_bringup/worlds/` or an absolute path |
| `robots_file` | `config/robots.yaml` | Robot names and spawn poses. Add entries to spawn more robots. |
| `sensors` | `auto` | Render sensors (gpu_lidar). `auto` turns them on with the GUI and off when headless. |

Per robot `<ns>` you get:

| ROS interface | notes |
|---|---|
| `/<ns>/cmd_vel` | `geometry_msgs/Twist`, ROS → Gazebo |
| `/<ns>/odom` | `nav_msgs/Odometry` with frames `<ns>/odom` → `<ns>/base_link` |
| `/<ns>/joint_states`, `/<ns>/robot_description` | robot_state_publisher `/<ns>/robot_state_publisher` |
| `/<ns>/scan`, `/<ns>/imu/data` | lidar only publishes when render sensors are enabled |
| tf | `world → <ns>/odom` (static, at the spawn pose) `→ <ns>/base_link → <ns>/chassis_link → …` on the global `/tf` |

## GPU options

| Mode | How to start | When to use it |
|---|---|---|
| **Intel/AMD (default)** | `docker compose up -d` or `./scripts/up.sh dri` | Mesa GPUs. `/dev/dri` is passed through for direct rendering. If there is no usable device, Mesa falls back to llvmpipe on its own. |
| **NVIDIA** | `docker compose -f docker-compose.yml -f docker-compose.nvidia.yml up -d` or `./scripts/up.sh nvidia` | Proprietary NVIDIA driver with nvidia-container-toolkit. `docker info` must list the `nvidia` runtime. This mode also enables PRIME render offload, so it works on hybrid laptops. |
| **Software** | `docker compose -f docker-compose.yml -f docker-compose.software.yml up -d` or `./scripts/up.sh software` | Anything else (VMs, broken drivers). Sets `LIBGL_ALWAYS_SOFTWARE=1`. It is slow but works everywhere. |

`./scripts/up.sh` with no argument picks NVIDIA if `nvidia-smi` works and Docker
has the nvidia runtime. Otherwise it uses DRI if `/dev/dri` exists, and software
rendering if not. To switch modes, run `docker compose down`, then start again
in the other mode. Build volumes are kept, so no rebuild happens.

To check what the container actually renders with:
`docker compose exec sim glxinfo -B | grep renderer`.

## Rebuilding

- **Code edits under `src/`** are visible in the container immediately (bind mount).
  - Python and launch files in `rover_multi_bringup` are symlink-installed, so edits take effect without rebuilding. New files still need a build.
  - C++ changes, or new packages, need a build: `./scripts/build.sh`. To build one package, use `./scripts/build.sh --packages-select <pkg>`.
- **Clean rebuild of the workspace:** `docker compose down -v && docker compose up -d`. Alternatively, run `FORCE_BUILD=1 docker compose up -d --force-recreate`.
- **Rebuild the image** after changing the Dockerfile or any `package.xml` dependency: `docker compose build`, then `docker compose up -d`.
- **Headless self-test:** `./scripts/smoke_test.sh`. It launches both robots without a GUI, then checks:
  - every expected topic and node exists;
  - odometry frames are correct;
  - every tf frame has exactly one parent and carries a robot prefix;
  - driving robot1 moves only robot1, and driving robot2 moves only robot2.

  It exits with status 0 on success.

## Layout

```
Dockerfile, docker-compose.yml          image + base (Intel/AMD) runtime config
docker-compose.nvidia.yml / .software.yml   GPU overrides
scripts/                                xhost, up, build, sim, teleop, smoke_test, entrypoint
src/roverrobotics_ros2/                 vendor repo, humble @ e6104d0, UNMODIFIED (git subtree)
src/rover_multi_bringup/                our package
  rover_multi_bringup/description.py    namespaces the vendor URDF per robot
  rover_multi_bringup/smoke_check.py    headless verification node
  launch/  gazebo | spawn_mini | single_mini | multi_mini .launch.py
  worlds/arena.sdf, config/robots.yaml, rviz/multi_mini.rviz
```

## Design decisions

- **Gazebo Fortress, not Classic.** The vendor package declares `ros_gz_sim` and `ros_gz_bridge`, and its URDF uses `ignition-gazebo-*` system plugins.
- **Namespacing method.** The Mini is driven by Gazebo system plugins with hard-coded absolute topics (`/cmd_vel`, `/odometry/wheels`, `/joint_states`, `scan`, …) and unprefixed frames. With no ros2_control, there is nothing to remap on the ROS side. Instead, `description.py` runs xacro on the **unmodified** vendor `mini.urdf` and makes three edits:
  1. It moves every plugin and sensor `*topic` under `/<ns>/`. DiffDrive odometry becomes `/<ns>/odom`.
  2. It prefixes every frame id with `<ns>/`.
  3. It removes the world-level Sensors and Imu systems that the vendor attaches to the model. They must exist once per world, not once per robot, so `arena.sdf` loads them.

  robot_state_publisher uses `frame_prefix: <ns>/`. DiffDrive's odom→base_link tf is bridged onto the global `/tf`. All robots share one `/tf` with prefixed frames, not a separate `/<ns>/tf` each, so a single RViz or tf listener sees every robot.
- **A shared `world` frame.** Each robot gets a static `world → <ns>/odom` transform at its spawn pose. DiffDrive odometry starts at zero where the robot spawns, so this places both robots correctly in RViz.
- **World.** `arena.sdf` is a flat 30×30 m plane with three static obstacles. The vendor worlds (`maze.sdf`, `depot.sdf`, `warehouse.sdf`) also work, for example `world:=$(ros2 pkg prefix roverrobotics_gazebo)/share/roverrobotics_gazebo/worlds/maze.sdf`. They don't load the Sensors or Imu systems, though, so lidar and IMU stay silent in them.
- **Physics step stays at 1 ms.** The world uses the same step size as the vendor worlds. Measured headless on an i7 laptop with two robots, the real-time factor was 0.41 at 1 ms. A 2 ms step reached 0.79, but odometry went wrong (x went negative under a forward command). A 4 ms step made the wheel mesh contacts unstable, and RTF fell to 0.07. With GUI and RViz, RTF was about 0.25 on the Intel iGPU, 0.29 on the RTX 5060, and 0.31 on llvmpipe. So the limit is CPU physics, not rendering. Phase 0 accepts that; later phases can revisit it, for example with simplified collision shapes in our own URDF wrapper.
- **Headless mode disables render sensors by default.** Offscreen (EGL) rendering picks the first DRM device. On hybrid laptops that can be a GPU with no usable driver in the container, and then gpu_lidar crashes the server. Driving, odometry, tf and the IMU need no rendering, so the headless smoke test is independent of the hardware. Force the lidar on with `sensors:=true`.
- **GPU default = `/dev/dri`.**
  - It works for Intel and AMD and degrades to llvmpipe on its own.
  - It is bind-mounted with a cgroup rule for DRM devices, not listed under `devices:`. As a result, the container still starts on hosts with no `/dev/dri`.
  - NVIDIA is opt-in, because it needs host software that not every machine has.
- **Container runs as root.** Build artifacts (`build/`, `install/`, `log/`) live in named Docker volumes, not in the repo. This avoids root-owned files in your checkout on every distro.
- **DDS isolation.** `ROS_LOCALHOST_ONLY=1` and `ROS_DOMAIN_ID=0` are the defaults, so teammates on the same LAN don't see each other's robots. Override them in a `.env` file next to `docker-compose.yml`, for example `ROS_LOCALHOST_ONLY=0` for multi-machine setups later.
- **Reproducibility / pinning.** The base image is pinned by **digest**, and the build never runs `apt-get upgrade`, so the ROS and Gazebo stack comes frozen from the image. Only three things install at build time: `teleop_twist_keyboard`, `mesa-utils`, and the rosdep keys (already present in the base). Those are not version-pinned, because packages.ros.org keeps only the latest sync, and an exact pin would break the build after the next ROS sync. The exact versions that went into your image are recorded in `/opt/image-packages.lock`. To compare two machines, run `docker compose exec sim cat /opt/image-packages.lock` on each and diff the output.
- **Vendor code as a git subtree** (squashed). A fresh clone is complete without any submodule step, and upstream can be pulled with `git subtree pull --prefix src/roverrobotics_ros2 https://github.com/RoverRobotics/roverrobotics_ros2 humble --squash`.

## Troubleshooting

**`DISPLAY is not set` from `docker compose`, or `could not connect to display`.**
Run the commands from a terminal inside your graphical session, not over plain
SSH. If you use `sudo docker compose`, DISPLAY is dropped. Run as a user in the
`docker` group instead, or use `sudo -E`.

**`Authorization required, but no authorization protocol specified`.**
The xhost step is missing. It is reset at every login. Run
`./scripts/xhost_setup.sh`, then launch again.

**The GUI doesn't appear but the launch keeps running.**
- Check `echo $DISPLAY` on the host. Then check that `/tmp/.X11-unix/X0` exists (use the number that matches your display).
- Test a trivial app: `docker compose exec sim glxgears`.
- On pure Wayland without XWayland, install and enable XWayland.
- If Gazebo opens as a black or empty window, try software mode (below) to rule out the driver.

**`OpenGL 3.3 is not supported`, `failed to create dri2 screen`, or `Failed to create OpenGL context`.**
Gazebo's Ogre2 renderer needs GL 3.3 or newer. Check what you get with
`docker compose exec sim glxinfo -B`.
- Intel/AMD: the host must load a Mesa kernel driver (`ls /dev/dri` should list `renderD128`).
- NVIDIA: use the NVIDIA mode. Mesa in the container can't drive NVIDIA's proprietary kernel driver.
- As a last resort, use `./scripts/up.sh software`.

**NVIDIA: `could not select device driver "nvidia"` or `unknown or invalid runtime`.**
Install nvidia-container-toolkit, then run
`sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`.
**Driver or library version mismatch** (`Failed to initialize NVML`, GLX errors
after a driver update): reboot the host so the kernel module and userspace
match, then restart the container. The toolkit injects the host's driver
libraries, so the image needs no NVIDIA driver.

**The sim is slow, or the real-time factor shown bottom-right in Gazebo is well below 100 %.**
- The vendor model uses full mesh collisions, which makes physics costly.
- Close RViz, or run headless with `gui:=false`.
- A faster GPU won't help much here: the bottleneck is CPU physics. In software mode RViz itself becomes sluggish.
- The teleop still works; the robots just move in sim-time.

**`ros2 topic list` on the host doesn't show the robots.**
The container uses `ROS_LOCALHOST_ONLY=1` and `ROS_DOMAIN_ID=0`. Match both on
the host, and use the same RMW implementation (the default is Fast DDS).

**The first `docker compose up` seems stuck.**
It is running `colcon build`. Watch it with `docker compose logs -f sim`.

## NEXT PHASES

The workspace is laid out so new work lands as new packages next to
`rover_multi_bringup`, with no restructuring:

- **Controller package**, for example `src/rover_controller/`. It subscribes to
  `/<ns>/odom` (plus `/<ns>/scan` and `/<ns>/imu/data`) and publishes
  `/<ns>/cmd_vel`. It replaces the teleop per robot. Start one instance per
  robot namespace from a new launch file that includes
  `multi_mini.launch.py`, or extend `config/robots.yaml` with a per-robot
  controller entry.
- **Communication-interface package**, for example `src/rover_comm_interface/`.
  It sits between the controller and `/<ns>/cmd_vel`, or between the robots, to
  model the network and transmission policy. A clean way to insert it is to
  remap the controller's output to `/<ns>/cmd_vel_request` and have this node
  forward to `/<ns>/cmd_vel`. Beyond one machine, set `ROS_LOCALHOST_ONLY=0`
  and choose a `ROS_DOMAIN_ID` in `.env`.
- Add their dependencies to each package's `package.xml`. The next
  `docker compose build` resolves them through rosdep.
