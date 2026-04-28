# reactive_replanning_ur12e

Reactive motion replanning for a UR12e + Robotiq Hand-E using a RealSense D435i for live obstacle detection. Built on ROS 2 Jazzy and MoveIt 2.

The arm runs a pick-and-place demo and routes around obstacles seen by the depth camera in real time. If a hand or other object enters the planned path mid-motion, the controller cancels cleanly, waits for the arm to settle, and re-plans from the current state.

## What's in here

```
launch/reactive_replanning_full.launch.py   # full stack: bringup + MoveIt + RViz + scene
config/reactive_replanning.rviz             # auto-loaded RViz with PointCloud2 + EEF path
config/sensors_3d.yaml                      # OctoMap point-cloud updater config
reactive_replanning_ur12e/
  reactive_replanning.py                    # main detection + replanning node
  scene_setup.py                            # static collision objects (floor)
  insert_obstacle.py                        # CLI helper to drop a sphere mid-demo
```

## Detection pipeline

Each cloud frame from `/camera/camera/depth/color/points` is filtered in this order:

1. **Depth filter** in camera frame (`DEPTH_MIN`–`DEPTH_MAX` meters)
2. **Batch TF transform** to `base_link` using a single rotation matrix multiply
3. **Workspace bounding box** in `base_link` — drops floor, walls, robot pedestal, anything outside the working volume
4. **Color filter** — drops UR12e's palette (light gray / off-white housing, black accents, light-blue/teal trim) with a Kovac-rule skin protection so hands never get filtered
5. **Multi-link robot self-filter** — line-segment distance to every link shaft + joint-origin spheres + tool0 gripper guard, all dilated across the last 3 frames of arm motion (kills phantoms from points that get unmasked when the arm moves past)

Whatever survives is foreign. Centroid → published as a `CollisionObject` sphere on `/apply_planning_scene` so MoveIt can plan around it.

## Replanning strategy

Primary: `MoveGroup` action with pose constraints + `replan=True`, using **BITstar** (asymptotically optimal OMPL planner) with a 2.5 s budget for smooth, near-optimal paths.

During execution a monitor runs at 10 Hz:
- Reads `tool0` from TF (no FK service call)
- If the obstacle has *moved* since plan time AND any upcoming waypoint is within `SPHERE_RADIUS + PATH_CLEARANCE` of it, preempts the goal
- Cancels cleanly, polls joint velocities until the arm is at rest, then recursively replans from the new state (capped at depth 2)

Fallback chain when the primary plan can't be found:
1. `MoveGroup` retry after clearing the sphere (handles `START_STATE_IN_COLLISION`)
2. **IK redundancy**: ~120 IK solutions, sorted by joint distance from current state, plan up to 8 candidates and execute the shortest under the trajectory-length cap. Uses RRTConnect for speed.
3. Arc detour (cartesian over the obstacle) as last resort

## Visualization

The launch file points RViz at `config/reactive_replanning.rviz`, which loads on startup with:

- **RobotModel** — live joint state
- **PointCloud2** — `/camera/camera/depth/color/points` (Best Effort QoS so it actually displays)
- **PlannedEEFPath** — `nav_msgs/Path` republish of MoveIt's `/display_planned_path`, drawn as a green line tracing the tool0 path
- **Trajectory** — animated robot ghost following the planned trajectory with a trail
- **MotionPlanning** — full MoveIt panel for manual planning

The path republisher subscribes to `/display_planned_path`, FKs each subsampled waypoint to get tool0, and publishes on `/planned_eef_path`. Skipped during IK redundancy to avoid choking the FK service queue.

## Running

Three terminals.

**T1 — full stack:**
```bash
ros2 launch reactive_replanning_ur12e reactive_replanning_full.launch.py
```

**T2 — RealSense camera with point cloud:**
```bash
ros2 launch realsense2_camera rs_launch.py pointcloud.enable:=true
```

**T3 — the demo:**
```bash
ros2 run reactive_replanning_ur12e reactive_replanning
```

When prompted, press ENTER to start the cycles. Move your hand into the workspace at any time to trigger detection and a replan.

## Tuning knobs

All exposed as ROS parameters on the demo node:

| Parameter         | Default | What it controls                                                         |
|-------------------|---------|--------------------------------------------------------------------------|
| `pick_z_offset`   | -0.25   | Pick depth below home (meters)                                           |
| `place_y_offset`  |  0.75   | Place horizontal offset from home                                        |
| `vel_scale`       |  0.10   | Joint velocity scaling                                                   |
| `acc_scale`       |  0.08   | Joint acceleration scaling (slightly < velocity for smoother ramps)      |
| `ik_attempts`     |  120    | IK solutions to generate for the redundancy pool                         |
| `ik_pool_size`    |  8      | How many IK candidates to plan before picking the shortest               |
| `max_traj_length` |  5.0    | Joint-space length (rad) above which an IK plan gets rejected            |
| `short_path_eps`  |  3.5    | First IK plan shorter than this exits the search early                   |

Class-level constants for filter geometry are at the top of `ReactiveReplannerUR12e` in `reactive_replanning.py` (`WS_X/Y/Z_MIN/MAX`, `ARM_SEG_RADIUS`, `ARM_JOINT_RADIUS`, `EEF_GUARD_RADIUS`, `SPHERE_RADIUS`, `PATH_CLEARANCE`, color thresholds).

## Notes

- The `ur12e_hande_bringup` package is treated as read-only — none of its files are touched. All custom config lives in this package.
- OctoMap's `occupancy_map_monitor` plugin doesn't load on this MoveIt 2 Jazzy build. Direct point-cloud → `CollisionObject` injection is used instead, which gives MoveIt's planning scene the same information without depending on the broken plugin.
- Velocity scaling is intentionally conservative. The UR controller's max deceleration is what actually triggers protective stops on cancel — ramping deceleration limits in the URDF/controllers config is the way to soften that further.
