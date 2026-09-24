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

Each teleop is a `teleop_twist_keyboard` publishing to `/<robot>/cmd_vel`.
**The teleop terminal itself must have keyboard focus** -- clicking the Gazebo
or RViz window sends your keys there instead, and the robot will not move.
**Arrow keys do nothing.** The keys are:

| key | action |
|---|---|
| `i` / `,` | forward / backward |
| `j` / `l` | turn left / right |
| `u` `o` `m` `.` | diagonal (forward-left, forward-right, back-left, back-right) |
| `k` | stop |
| `q` / `z` | faster / slower (all), `w`/`x` linear only, `e`/`c` turn only |
| `Ctrl-C` | quit |

One press is enough: the command keeps applying until you press another key,
so `i` drives until you press `k`. If you prefer not to use the keyboard:

```bash
./scripts/drive.sh robot1              # forward 0.5 m/s for 3 s, prints odom
./scripts/drive.sh robot2 0.4 0.5 5    # linear.x angular.z seconds
```

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

## Gazebo vs RViz: which window shows what

They are not two views of the same thing, and you do not need both.

| | Gazebo | RViz |
|---|---|---|
| What it is | the simulator: physics, the world, ground truth | a viewer for ROS topics |
| Shows the obstacles and ground plane | yes, they live in `arena.sdf` | **no** -- nothing publishes them as ROS messages |
| Shows the robots | ground-truth pose | pose from `/tf` + `/<ns>/odom`, model from `robot_description` |
| Shows lidar, odometry, tf | only as optional overlays | yes, this is the point of it |
| Needed to run the sim | yes (can be headless) | no |

**Obstacles are missing in RViz because RViz only knows what the robots
publish.** The orange and blue point arcs *are* those obstacles, seen by each
robot's lidar. That is the view later phases care about: what a robot knows,
rather than what is true.

Three workflows, pick one:

```bash
./scripts/sim.sh                             # both windows (default)
./scripts/sim.sh rviz:=false                 # Gazebo only: watch the world, lightest to understand
./scripts/sim.sh gui:=false sensors:=true    # RViz only: headless physics, nicer robot rendering
```

The RViz-only mode runs the simulator with no Gazebo window and renders the
lidar offscreen, so you still see the obstacle outlines. On this machine it
ran at 31 fps with a slightly better real-time factor than the Gazebo GUI.
It needs working offscreen (EGL) rendering: verified on the NVIDIA path, and
it works on Intel/AMD where the first DRM device is the rendering one. If the
scans stay empty, drop `sensors:=true` and you still get robots, tf and
odometry.

If you want the obstacles themselves drawn in RViz, that takes a small node
publishing them as markers. Ask and I can add one.

## GPU options

| Mode | How to start | When to use it |
|---|---|---|
| **Intel/AMD (default)** | `docker compose up -d` or `./scripts/up.sh dri` | Mesa GPUs. `/dev/dri` is passed through for direct rendering. If there is no usable device, Mesa falls back to llvmpipe on its own. |
| **NVIDIA** | `docker compose -f docker-compose.yml -f docker-compose.nvidia.yml up -d` or `./scripts/up.sh nvidia` | Proprietary NVIDIA driver with nvidia-container-toolkit. `docker info` must list the `nvidia` runtime. This mode also enables PRIME render offload, so it works on hybrid laptops, and points offscreen (EGL) rendering at the NVIDIA GPU so headless lidar works. |
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
scripts/                                xhost, up, build, sim, teleop, drive, watch_cmd_vel, smoke_test, entrypoint
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
- **Headless mode disables render sensors by default.** Offscreen (EGL) rendering picks the first DRM device. On a hybrid laptop that can be a GPU that Mesa cannot drive in the container, and then gpu_lidar crashes the server. Driving, odometry, tf and the IMU need no rendering, so the headless smoke test is independent of the hardware. Force the lidar on with `sensors:=true`; on NVIDIA the image supplies the glvnd EGL vendor file the container toolkit omits, and the NVIDIA override restricts EGL to that GPU, which makes headless lidar work.
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

**I press keys in teleop but the robot doesn't move.**
Work through this in order:
1. Check the sim itself: `./scripts/drive.sh robot1`. If the robot moves in Gazebo, the simulator and the whole command path are fine, and the problem is the keystrokes.
2. Click the **teleop terminal** so it has focus. Keys typed into the Gazebo or RViz window never reach teleop.
3. Use `i`, `j`, `k`, `l`, `,` -- **not the arrow keys**. Any unbound key sends a stop command, so arrow keys hold the robot still.
4. Watch what teleop actually sends: run `./scripts/watch_cmd_vel.sh robot1` in a third terminal, then press `i` in the teleop terminal. Nothing printed means keystrokes aren't arriving (focus or key problem). Printed `0.5` while the robot stays put means look at the sim instead.
5. Movement is genuinely slow: the sim runs at roughly 0.3x real time, so 0.5 m/s looks like about 0.15 m/s on screen. Give it a few seconds, or speed up with `q`.
6. Teleop must run in a real terminal. Starting it from an editor's "run" button or a pipe gives it no TTY, and it cannot read keys.

**`ros2 topic list` on the host doesn't show the robots.**
The container uses `ROS_LOCALHOST_ONLY=1` and `ROS_DOMAIN_ID=0`. Match both on
the host, and use the same RMW implementation (the default is Fast DDS).

**The first `docker compose up` seems stuck.**
It is running `colcon build`. Watch it with `docker compose logs -f sim`.

## Phase 1: the task and the evaluation harness

Phase 1 adds the *measurement ruler*: a leader-follower task whose information
demand varies over time, four baseline transmission policies, and the metrics
and figures every later controller is judged by. No learning yet -- but the
interfaces an RL policy and a spiking network plug into are fixed now.

### The task

`robot1` (leader) tracks a fixed closed path with pure pursuit and carries
small bounded process noise. `robot2` (follower) holds a slot 0.8 m directly
behind it, in the leader's body frame.

The point is the middle step: **the follower never sees the leader's true
state.** A communication interface decides, each step, whether the leader
transmits. When it does, the follower's estimate is refreshed; when it does
not, a *generic* constant-velocity predictor dead-reckons it forward. The
predictor knows nothing about the path, so:

| leader is... | predictor | messages needed |
|---|---|---|
| on a straight | near-exact | almost none |
| in a steady turn | still exact (constant curvature) | almost none |
| entering/leaving a turn, or pushed by noise | wrong, and drifting | many |

That variation is deliberate and it is what a learned scheduler will exploit.

### Two backends, one interface

```
                        formation_core (pure Python, no ROS, no RL)
                        contract | policies | predictor | metrics | runner
                                        |
              +-------------------------+-------------------------+
              |                                                   |
    FastFormationEnv (kinematic twin)              GazeboFormationEnv + ROS nodes
    thousands of steps/second, for training        Phase 0 /robot1, /robot2 topics
```

The same controller, policy and metrics code runs on both. Measured on the same
config (20 s, event-triggered delta = 0.05):

| | formation RMS | lateral RMS | leader path RMS | comm rate |
|---|---|---|---|---|
| fast twin | 0.0619 m | 0.0529 m | 0.0089 m | 0.018 |
| Gazebo | 0.0573 m | 0.0523 m | 0.0088 m | 0.013 |

### Running it

All commands run inside the container (`docker compose exec sim bash`). The
fast twin needs no simulator.

```bash
cd /ws/src/formation_core

# one episode + figures  (results/episode/: episode.csv, errors.png, trajectory.png)
python3 -m formation_core run --policy event_triggered --delta 0.05 --plot

# the 8-seed evaluation suite, mean +- 95% CI across seeds
python3 -m formation_core suite --suite configs/eval_suite.yaml
python3 -m formation_core suite --suite configs/stress_suite.yaml --policy periodic --k 10

# THE key artifact: the Pareto sweep over all three families, with the check
python3 -m formation_core sweep --check --out results/sweep

# unit tests (74)
python3 -m pytest tests -q
```

### Figures for a talk

`formation_core figures` writes the five slide-sized figures that carry the
result, as PNG (slides) and PDF (print), from the same data and the same
palette as the paper figures:

```bash
python3 -m formation_core sweep --out results/sweep          # figure 2 needs this
python3 -m formation_core figures --out results/figures \
    --sweep results/sweep \
    --gazebo-results results/gz_policies \
    --gazebo-config src/formation_gazebo/config/gazebo_episode.yaml
```

| | Figure | The one question it answers |
|---|---|---|
| 1 | `01_matched_budget` | At a fixed message budget, how much better is event-triggering? |
| 2 | `02_pareto` | Does that hold across the whole budget range? |
| 3 | `03_mechanism` | *Why* does it win -- where does each policy spend its messages? |
| 4 | `04_trajectories` | What does the failure look like on the ground? |
| 5 | `05_sim_to_sim` | Does the fast twin's answer survive a real simulator? |

`--only budget pareto mechanism trajectories transfer` regenerates a subset.
Figures 1, 3 and 4 run their own episodes; figure 2 reads `pareto.json` from the
sweep, and figure 5 reads Gazebo run directories, taking each run's **own**
saved `config.yaml` so the twin re-runs the identical episode.

Figure 4 colours the follower's track by formation error rather than drawing
leader and follower tracks: most of the error is along-track, so a 0.05 m and a
0.30 m episode trace nearly the same oval and plain tracks show nothing.

`python3 -m formation_core <command>` works everywhere. The same entry points
are installed as console scripts, reachable as
`ros2 run formation_core formation-sweep` in the ROS workspace. Outside ROS the
package is an ordinary pip install:

```bash
pip install -e src/formation_core     # numpy + pyyaml; matplotlib for figures
```

In Gazebo (needs the Phase 0 sim, which the launch file starts for you):

```bash
ros2 launch formation_gazebo formation.launch.py                        # GUI
ros2 launch formation_gazebo formation.launch.py policy:=periodic k:=10
./scripts/formation_smoke.sh                                            # headless check
```

`formation.launch.py` spawns the robots on the path in an empty arena, then
starts the leader, comm-interface, controller and evaluation nodes. The
evaluation node writes `episode.csv` and `metrics.csv` with the same columns
and metric definitions as the fast twin, then exits.

### The result Phase 1 exists to establish

`formation_core.sweep --check` sweeps periodic(k), random(p) and
event_triggered(delta) over the 8-seed evaluation suite and asserts the premise.
It passes on both the evaluation and stress suites:

* error rises as the communication rate falls, for every family;
* **event-triggered beats periodic and random at every matched rate** -- by 82%
  and 93% respectively at the low-rate end, converging as the rate approaches 1.

That is what makes the task worth learning on: at a fixed message budget,
*when* you transmit matters, so there is something for an RL/spiking scheduler
to discover. The same ordering reproduces in Gazebo (event-triggered: 0.057 m
with 5 messages; periodic k=20: 0.061 m with 20 messages).

### The frozen contract

`formation_core/contract.py` is the one file later phases must not break.

**Version 2.0 (Phase 2).** One centralized controller drives both robots, so
the observation covers both and the action commands both: **16 observations, 4
actions**, all normalized to [-1, 1] and all derived from the per-robot
ESTIMATES, never ground truth.

| idx | name | meaning |
|---|---|---|
| 0, 1 | `lead_look_x/y` | path lookahead point in **leader** body frame / `max_range` |
| 2, 3 | `lead_sin/cos_tangent` | path tangent there, relative to leader heading |
| 4, 5 | `lead_v`, `lead_w` | estimated leader velocities / limits |
| 6 | `lead_age` | steps since the leader's estimate was refreshed / `age_max` |
| 7, 8 | `foll_v`, `foll_w` | estimated follower velocities / limits |
| 9 | `foll_age` | steps since the follower's estimate was refreshed / `age_max` |
| 10, 11 | `slot_ex/ey` | slot position in **follower** body frame / `max_range` |
| 12, 13 | `rel_dx/dy` | leader position in follower body frame / `max_range` |
| 14, 15 | `rel_sin/cos_dtheta` | leader heading relative to follower |
| action 0, 1 | `lead_v`, `lead_w` | leader command, scaled to +-`v_max`, +-`w_max` |
| action 2, 3 | `foll_v`, `foll_w` | follower command, same scaling |

Angles appear only as (sin, cos), so there is no wraparound for a network to
model; everything is relative, so a policy cannot memorise the path;
`observation_space` / `action_space` are Box objects mirroring Gymnasium, and
`reset()`/`step()` already return Gymnasium's tuples, so a Gym wrapper is a
thin adapter. `CONTRACT_VERSION` is recorded in every result file.

**Why the two age fields are mandatory.** Under perfect communications they read
~0 every step and carry no information, so nothing in Phase 2 would notice them
missing. They are the single point where the control half of the project touches
the communication half: once Phase 3 adds delay and loss and Phase 4 learns a
per-robot transmission policy, the controller has to know how stale each
estimate is to act cautiously on an old one. Adding them later would invalidate
every policy trained before, so they cost two inputs now.

**What changed from v1.0, and why it is a re-freeze rather than a widening.**
v1.0 was follower-shaped: 10 observations, 2 actions, and deliberately *no* path
information, on the rule that the follower only had to chase the leader. A
centralized controller drives the leader, so it needs a path reference. Old
v1.x CSVs and figures are untouched and stay readable -- they record
`contract_version` 1.0.

### Judgment calls (defaults, and why)

| Choice | Default | Reasoning |
|---|---|---|
| Path | oval: 4 m straights + 1.5 m radius caps | Zero curvature next to constant curvature is the sharpest contrast in predictor difficulty. `figure8` is also available. |
| Process noise | bounded AR(1), +-0.05 m/s, +-0.15 rad/s, rho 0.9 | Correlated, not white: white noise averages out and a CV predictor barely notices it. Bounded so the leader stays well behaved. |
| Offset `d` | 0.8 m behind | Far enough that estimate error matters, close enough to stay in sensor range later. |
| `dt`, duration | 0.05 s (20 Hz), 60 s | 20 Hz matches the Gazebo adapter; 60 s is about two laps. |
| Normalization | `max_range` 3 m, `v_max` 1 m/s, `w_max` 2 rad/s | Tight ranges give population encoders better resolution; values saturate rather than escape [-1, 1]. |
| Event threshold | position error only (`heading_weight` = 0) | Makes `delta` read directly as metres. Heading can be folded in via config. |
| Sweep ranges | k in 1..100, p in 1..0.01, delta in 0..0.3 m | Chosen so all three families span the same 0.01-1.0 rate range, which is what makes matched-rate comparison possible. |
| Aggregation | mean +- 95% CI, Student-t, 8 seeds | t(7) = 2.365, not 1.96; with 8 seeds the normal approximation understates the interval. |
| Error sign convention | errors point from follower TO slot | `longitudinal` > 0 means lagging; `lateral` > 0 means the slot is to the follower's left. |
| `age_max` (v2.0) | 50 steps = 2.5 s | Where the age inputs saturate. Past ~2.5 s a constant-velocity prediction of this leader is worthless, so the controller gains nothing from distinguishing degrees of "hopelessly stale". |
| `lookahead_distance` (v2.0) | 0.6 m, fixed | The observation's lookahead does NOT scale with speed, unlike `PurePursuitLeader`'s internal one: a learned policy should not have the meaning of its inputs change with the robot's speed. 0.6 m matches what the scripted leader used at its 0.6 m/s cruise. |
| `w_path` (v2.0) | 1.0 | Keeping the leader on the path became the controller's job when the scripted leader was removed, so it has to be paid for. Weighted equal to formation error. |
| `w_progress` (v2.0) | 0.5, credited per step and clipped to [-1, 1] | **Centralization created a trivial optimum and this closes it.** Every other term is a penalty, so parking both robots on the path in perfect formation scores ~0 and beats driving. Under v1.x the scripted leader always drove and the follower had to keep up; taking the scripted leader away removed the thing forcing motion. PPO found this immediately: the first trained policy scored 52% "better" than analytic by covering 0.27 laps at 0.04 m/s. Credit is capped at one step of `leader.target_speed`, so there is no reward for speeding and the analytic controller's curvature slowdown is not punished. |
| `terminal_penalty` (v2.0) | 50.0 | The other trivial optimum: every other term is a penalty, so an episode that ends early accumulates *less* of it than one that holds formation to the end, and PPO learns to drive off the path on purpose. Charged once on divergence, never on truncation. |
| `w_comm` | 0.0 everywhere | Plumbed through as a count over both robots so Phase 4 turns it on with a config edit, not a code change. |
| Actor architecture | MLP, 2 x 64, tanh | Deliberately convertible to a population-coded spiking network: no recurrence, no attention, no normalization. 5.5k parameters. |
| PPO | 600k steps, 8 envs x 256 rollout, lr 3e-4 (linearly decayed), gamma 0.99, lambda 0.95, clip 0.2, 10 epochs x 8 minibatches | Ordinary CleanRL defaults, tuned only enough to beat the analytic baseline. Recorded in `results/rl/ppo_config.json`. |
| Training seeds | 64 episode seeds, cycled across workers | Each worker resets on a different seed so the policy sees many realisations of the leader's noise instead of memorising one. |

Two things worth knowing when reading the numbers:

* **Heading error is large by construction** (~0.4 rad RMS on the default oval)
  and is not a controller fault: a robot holding a slot 0.8 m behind on a
  1.5 m-radius curve is genuinely rotated relative to the leader. Judge
  formation quality by the position errors.
* **The leader's process noise enters differently in the two backends.** The
  fast twin perturbs realised velocities; Gazebo can only be perturbed through
  commands, which it then tracks through its own dynamics. Same effect on
  predictability, not step-identical.

### Layout

```
src/formation_core/                 pure Python, pip-installable, no ROS, no torch
  formation_core/contract.py        FROZEN obs/action contract v2.0  <- start here
  formation_core/env.py             FormationEnv + FastFormationEnv (two uplinks)
  formation_core/comm.py            comm interface + Channel hook (Phase 3)
  formation_core/policies.py        always | periodic | random | event_triggered
  formation_core/predictor.py       generic constant-velocity dead reckoning
  formation_core/controllers.py     Controller interface, AnalyticCentralized, laws
  formation_core/metrics.py         control + comm metrics, CSV, CI aggregation
  formation_core/sweep.py           Pareto sweep + premise check
  formation_core/figures.py         the five presentation figures
  configs/                          default | eval_suite | stress_suite
  tests/                            79 unit tests
src/formation_rl/                   PPO controller (torch + gymnasium live HERE)
  formation_rl/actor.py             the MLP actor  <- the ANN -> SNN swap point
  formation_rl/policy.py            RLController: wraps an actor as a Controller
  formation_rl/ppo.py               CleanRL-style PPO; imports the actor, does not define it
  formation_rl/gym_env.py           thin Gymnasium view of FastFormationEnv
  tests/                            14 unit tests
src/formation_gazebo/               ROS 2 nodes, importing formation_core
  formation_gazebo/ros_interface.py odom <-> RobotState, world-frame transforms
  formation_gazebo/leader_node.py   reference-path publisher (no longer drives)
  formation_gazebo/comm_interface_node.py   ONE PER ROBOT
  formation_gazebo/controller_node.py       ONE controller, both cmd_vels
  formation_gazebo/evaluation_node.py       records both uplinks
  formation_gazebo/env.py           GazeboFormationEnv (same interface as the twin)
  launch/formation.launch.py        sim + reference path + 2 comm + controller + eval
```

## Phase 2: one centralized controller

Phase 1 had two controllers: a scripted pure-pursuit leader and a learned-slot
follower. Phase 2 replaces both with **one controller that commands both
robots**, under perfect communications. The transmission policy stays
per-robot -- that is the research question, and it is not a property of the
controller.

```
                        ONE centralized controller
                  obs (16, both robots) -> action (4, both robots)
                                  ^                |
                     estimates    |                |  cmd_vel x2 (reliable)
                                  |                v
        +-------------------------+----------------+-------------------------+
        |                                                                    |
   robot1 comm interface                                        robot2 comm interface
   (own policy, predictor,                                      (own policy, predictor,
    estimate, RNG stream)                                        estimate, RNG stream)
```

Both robots report *up* to the coordinator, where the controller runs. The
downlink is assumed reliable; Phase 3 models the uplink and says so.

### What changed

| | v1.x | v2.0 |
|---|---|---|
| Controller | scripted leader + learned follower | **one**, drives both |
| Observation / action | 10 / 2 | **16 / 4** |
| Communication interfaces | 1 (leader -> follower) | **2**, one per robot, both -> coordinator |
| Path tracking | the scripted leader's job, free | the controller's job, paid for by `w_path` |
| Estimate staleness | not in the observation | **`lead_age`, `foll_age`** -- mandatory |
| `leader_node` | drove `/robot1` | publishes the reference path only |

### Running it

```bash
cd /ws/src/formation_core

# the analytic baseline, on the fast twin
python3 -m formation_core run --plot
python3 -m formation_core suite --suite configs/eval_suite.yaml

# train the PPO actor (~4 min on CPU), then benchmark it against analytic
python3 -m formation_rl train --out results/rl
python3 -m formation_rl benchmark --weights results/rl/actor.pt

# either controller, on the fast twin
python3 -m formation_core suite --controller analytic
python3 -m formation_core suite --controller rl
```

In Gazebo, the launch file starts the reference-path publisher, **two**
communication interfaces and the one controller:

```bash
ros2 launch formation_gazebo formation.launch.py                    # analytic
ros2 launch formation_gazebo formation.launch.py controller:=rl     # learned
./scripts/formation_smoke.sh                                        # headless check
POLICY=always ./scripts/formation_smoke.sh                          # perfect comms
```

### Results

Analytic centralized controller, perfect communications, mean +- 95% CI over
the 8 fixed seeds:

| suite | formation RMS (m) | formation max (m) | path RMS (m) |
|---|---|---|---|
| eval | 0.0450 +- 0.0003 | 0.0749 +- 0.0024 | 0.0103 +- 0.0004 |
| stress | 0.1085 +- 0.0006 | 0.1611 +- 0.0033 | 0.0178 +- 0.0004 |

In Gazebo, 500 steps with perfect communications: **0.0399 m** formation RMS
and 0.0082 m path RMS, against 0.0489 m and 0.0093 m for the v1.x split
controller. Centralizing did not cost accuracy; it slightly improved it,
because the leader is now steered from the same lookahead the follower is
holding station against.

PPO_RESULTS_PLACEHOLDER

### Where the next phases plug in

* **Spiking actor**: replace `formation_rl/actor.py` only -- see NEXT PHASES.
* **Per-robot learned transmission policy**: each `comm_interface_node` already
  owns its own policy instance; `reward.w_comm` is already plumbed and already
  counts both robots. The controller can already see staleness.
* **Network model**: `formation_core.comm.Channel` is untouched and still
  dormant.

## NEXT PHASES

The workspace is laid out so new work lands as new packages beside the existing
ones, with no restructuring. The interfaces they plug into already exist:

* **Spiking controller** (PopSAN-style, SpiNNaker) -- the next controller step.
  It replaces **one module**: `formation_rl/actor.py`. Implement `act(obs) ->
  action` on a population-coded network and nothing in training, evaluation,
  the ROS nodes or the metrics changes; `RLController(actor=...)` takes any
  object with that method, and `tests/test_actor_and_policy.py` already proves
  it with a stand-in. The MLP it replaces is constrained for exactly this:
  feedforward only, no recurrence, no attention, no normalization layers, tanh
  in and out, 5.5k parameters. The contract was sized for it too -- 16 inputs
  and 4 outputs, all in [-1, 1], angles as (sin, cos).
* **Learned transmission policy** (Phase 4), one **per robot**. Implement
  `formation_core.policies.TransmissionPolicy`; `PolicyContext` already carries
  what a scheduler needs (prediction error, age, timing). Each robot's
  `comm_interface_node` already owns its own policy instance and its own RNG
  stream, and `FastFormationEnv` already builds one interface per robot, so
  giving the two robots different policies is a config change. Turn on
  `reward.w_comm` (already plumbed, already counting both robots) and the
  learned policy appears in the Pareto plot next to the baselines. The
  controller can already see what it needs to cope: `lead_age` and `foll_age`
  are in the observation.
* **Network model** (Phase 3: delay, loss, jitter). Implement
  `formation_core.comm.Channel` (`send`/`deliver`) and pass it to
  `CommInterface`. Policies, controllers and metrics need no change --
  `tests/test_comm_policies.py` already exercises the hook with a delaying
  channel.
* **Phase 0 additions** (the original note): a controller package and a
  communication-interface package now exist as `formation_core` +
  `formation_gazebo`; multi-machine runs need `ROS_LOCALHOST_ONLY=0` and a
  shared `ROS_DOMAIN_ID` in `.env`.

Add each package's dependencies to its `package.xml`; the next
`docker compose build` resolves them through rosdep.
