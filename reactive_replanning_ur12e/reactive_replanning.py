#!/usr/bin/env python3
"""
Reactive Replanning for UR12e + Robotiq Hand-E
Adapted from ME5250 UR5 project by Abdul Rahman.

Strategy: Exploit kinematic redundancy — compute 30 IK solutions for the
goal pose, rank by manipulability, and iterate through them when an obstacle
blocks the current trajectory.
"""
print(''.join(chr(x-7) for x in [104,105,107,124,115,39,121,104,111,116,104,117]))

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from moveit_msgs.msg import CollisionObject, MoveItErrorCodes, Constraints, JointConstraint, RobotState
from moveit_msgs.msg import PositionConstraint, OrientationConstraint, BoundingVolume
from moveit_msgs.msg import PlanningScene as PlanningSceneMsg
from moveit_msgs.msg import DisplayTrajectory
from moveit_msgs.srv import GetPositionIK, GetPositionFK, GetPlanningScene, GetCartesianPath, ApplyPlanningScene
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from control_msgs.action import GripperCommand
from geometry_msgs.msg import Pose, PointStamped, PoseStamped
from sensor_msgs.msg import JointState, PointCloud2
from sensor_msgs_py import point_cloud2
from shape_msgs.msg import SolidPrimitive
from trajectory_msgs.msg import JointTrajectory
from nav_msgs.msg import Path
import tf2_ros
from tf2_geometry_msgs import do_transform_point

import numpy as np
import time
import threading
from collections import deque
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class IKSolution:
    joint_positions: List[float]
    manipulability: float
    is_valid: bool


class ReactiveReplannerUR12e(Node):
    def __init__(self):
        super().__init__('reactive_replanner_ur12e')

        self._executor: Optional[MultiThreadedExecutor] = None
        self._executing = False
        self.callback_group = ReentrantCallbackGroup()

        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint',
        ]

        # Offsets relative to home EEF pose — bigger = longer travel = more
        # time to detect/react before the arm hits the obstacle.
        self.declare_parameter('pick_z_offset',  -0.25)  # down from home
        self.declare_parameter('place_y_offset',  0.75)  # right of home — farther goal for more visible motion
        self.declare_parameter('ik_attempts',     120)
        self.declare_parameter('vel_scale',       0.10)
        self.declare_parameter('acc_scale',       0.08)
        self.declare_parameter('max_traj_length', 5.0)   # rad — reject IK plans longer than this
        self.declare_parameter('ik_pool_size',    8)     # how many IK candidates to plan before picking shortest
        self.declare_parameter('short_path_eps',  3.5)   # rad — first plan shorter than this wins immediately
        self._current_goal_pose: Optional[Pose] = None

        self.current_joint_state = None
        self.obstacle_detected = False
        self._cloud_baseline: Optional[int] = None

        # Computed from FK at startup
        self.home_eef_pose: Optional[Pose] = None
        self.pick_pose:     Optional[Pose] = None
        self.place_pose:    Optional[Pose] = None

        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self._js_cb, 10,
            callback_group=self.callback_group)

        self.cloud_sub = self.create_subscription(
            PointCloud2, '/camera/camera/depth/color/points', self._cloud_cb, 5,
            callback_group=self.callback_group)

        # Direct point-cloud obstacle detection (OctoMap plugin doesn't load on this build).
        self._camera_ok = False
        self._baseline_count: Optional[int] = None
        self._baseline_samples: List[int] = []
        self._obstacle_present = False
        self._last_cloud_t = 0.0
        self._obstacle_streak = 0           # consecutive frames over threshold
        self._last_obstacle_seen = 0.0      # for TTL-based removal
        self._last_obstacle_xyz = None      # base_link position of last sphere
        self._protective_stop = False       # latched if UR controller rejects a goal
        self.last_failure_code = 0          # last MoveIt error code from move_to_pose
        self._planned_eef_xyz: List[tuple] = []  # cached EEF (x,y,z) waypoints of latest plan
        self._skip_path_viz = False         # set True during IK redundancy to avoid FK spam
        # Recent arm joint positions for swept-volume self-filter. Without this,
        # points that *were* inside the arm self-filter become unmasked the
        # moment the arm moves past, registering as phantom obstacles for the
        # one frame before the baseline catches up. 3 frames at 20 Hz = ~0.15 s
        # of motion — enough to kill the transients without masking the hand
        # area for too long after the arm has cleared.
        self._arm_pos_history: deque = deque(maxlen=3)

        self._collision_pub = self.create_publisher(
            CollisionObject, '/collision_object', 10)

        # EEF path visualization: subscribe to MoveIt's DisplayTrajectory and
        # republish the tool0 path as nav_msgs/Path so RViz draws a clean line
        # (Nav2-style) for every plan it produces.
        self._eef_path_pub = self.create_publisher(Path, '/planned_eef_path', 5)
        self.display_traj_sub = self.create_subscription(
            DisplayTrajectory, '/display_planned_path',
            self._display_traj_cb, 1, callback_group=self.callback_group)

        # TF buffer to convert obstacle point from camera optical frame → base_link
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Real-time preemption state
        self._exec_lock = threading.Lock()
        self._current_gh = None     # active ExecuteTrajectory goal handle
        self._current_res_fut = None
        self._replan_in_progress = False

        self.ik_client = self.create_client(
            GetPositionIK, '/compute_ik', callback_group=self.callback_group)
        self.fk_client = self.create_client(
            GetPositionFK, '/compute_fk', callback_group=self.callback_group)
        self.cartesian_client = self.create_client(
            GetCartesianPath, '/compute_cartesian_path', callback_group=self.callback_group)
        self.scene_client = self.create_client(
            GetPlanningScene, '/get_planning_scene', callback_group=self.callback_group)
        self.apply_scene_client = self.create_client(
            ApplyPlanningScene, '/apply_planning_scene', callback_group=self.callback_group)

        self.move_client = ActionClient(
            self, MoveGroup, '/move_action', callback_group=self.callback_group)
        self.exec_client = ActionClient(
            self, ExecuteTrajectory, '/execute_trajectory', callback_group=self.callback_group)
        self.gripper_client = ActionClient(
            self, GripperCommand, '/gripper_action_controller/gripper_cmd',
            callback_group=self.callback_group)

        self.get_logger().info('Waiting for services...')
        self.ik_client.wait_for_service()
        self.fk_client.wait_for_service()
        self.cartesian_client.wait_for_service()
        self.scene_client.wait_for_service()
        self.apply_scene_client.wait_for_service()
        self.move_client.wait_for_server()
        self.exec_client.wait_for_server()
        self.gripper_client.wait_for_server()
        self.get_logger().info('All services ready — UR12e reactive replanner online.')

    def _spin(self, fut, timeout_sec=10.0):
        # Main thread executor is already spinning — just wait for the future.
        # Calling spin_until_future_complete from a non-executor thread raises RuntimeError.
        deadline = time.time() + timeout_sec
        while not fut.done() and time.time() < deadline:
            time.sleep(0.01)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _display_traj_cb(self, msg: DisplayTrajectory):
        """Whenever MoveIt publishes a planned trajectory, FK + cache + republish as Path.

        Skipped during IK redundancy planning (would spam 1000+ FK calls).
        """
        if self._skip_path_viz or not msg.trajectory:
            return
        traj = msg.trajectory[0].joint_trajectory
        if not traj.points:
            return
        self._publish_eef_path(traj)

    def _publish_eef_path(self, traj: JointTrajectory):
        """Compute FK for a subsampled joint trajectory, cache + publish as nav_msgs/Path."""
        n = len(traj.points)
        if n < 2:
            return
        step = max(1, n // 20)
        idxs = list(range(0, n, step))
        if idxs[-1] != n - 1:
            idxs.append(n - 1)

        path = Path()
        path.header.frame_id = 'base_link'
        path.header.stamp = self.get_clock().now().to_msg()
        cached_xyz = []

        joint_names = list(traj.joint_names)
        for i in idxs:
            req = GetPositionFK.Request()
            req.header.frame_id = 'base_link'
            req.fk_link_names = ['tool0']
            req.robot_state.joint_state.name = joint_names
            req.robot_state.joint_state.position = list(traj.points[i].positions)

            fut = self.fk_client.call_async(req)
            self._spin(fut, timeout_sec=0.1)
            res = fut.result()
            if res is None or res.error_code.val != MoveItErrorCodes.SUCCESS:
                continue
            p = res.pose_stamped[0].pose
            ps = PoseStamped()
            ps.header.frame_id = 'base_link'
            ps.pose = p
            path.poses.append(ps)
            cached_xyz.append((p.position.x, p.position.y, p.position.z))

        if path.poses:
            self._eef_path_pub.publish(path)
            # Cache for mid-execution path-vs-sphere intersection check
            self._planned_eef_xyz = cached_xyz

    def _traj_length(self, traj: Optional[JointTrajectory]) -> float:
        """Sum of joint-space distances between consecutive trajectory waypoints (radians)."""
        if traj is None or len(traj.points) < 2:
            return float('inf')
        total = 0.0
        for i in range(1, len(traj.points)):
            d = 0.0
            for a, b in zip(traj.points[i].positions, traj.points[i-1].positions):
                d += (a - b) ** 2
            total += d ** 0.5
        return total

    def _make_pose(self, x, y, z, qx, qy, qz, qw) -> Pose:
        p = Pose()
        p.position.x, p.position.y, p.position.z = float(x), float(y), float(z)
        p.orientation.x, p.orientation.y = float(qx), float(qy)
        p.orientation.z, p.orientation.w = float(qz), float(qw)
        return p

    def _js_cb(self, msg: JointState):
        self.current_joint_state = msg

    DEPTH_MIN = 0.4    # m, in camera optical frame
    DEPTH_MAX = 2.5    # m
    OBSTACLE_THRESHOLD = 120   # workspace foreign points to declare an obstacle (after TF + ws + arm + color filter)
    # Color filter: drop points whose color matches the robot (light gray/white)
    # or pure black (cables / accents). Anything else — skin, clothing, colored
    # objects — survives. Combined with the geometric self-filter, this kills
    # almost all phantom arm detections.
    USE_COLOR_FILTER  = True
    COLOR_GRAY_SUM    = 400    # if R+G+B exceeds this AND channels are similar → light gray
    COLOR_GRAY_DIFF   = 35     # max channel difference to call it "gray"
    COLOR_BLACK_SUM   = 90     # if R+G+B below this → black
    SPHERE_RADIUS = 0.12       # 12 cm — sized to actually disrupt straight-line paths
    # Preempt fires when a planned waypoint is within (SPHERE_RADIUS + PATH_CLEARANCE)
    # of the sphere center. OMPL already keeps the path SPHERE_RADIUS clear of obstacles
    # at plan time, so we use a TIGHT clearance here — preempt only when the sphere has
    # moved INTO the path (closer than OMPL's plan-time clearance).
    PATH_CLEARANCE = 0.02      # m — small safety buffer beyond sphere surface
    # Workspace bounding box in base_link frame. Detections OUTSIDE this box are
    # ignored — that's the camera seeing the robot's own pedestal, the floor,
    # background clutter, or the user's body. Only the volume in front of the
    # robot where the arm actually operates counts as "obstacles".
    WS_X_MIN, WS_X_MAX = -1.10, -0.30
    WS_Y_MIN, WS_Y_MAX = -0.45,  0.55
    WS_Z_MIN, WS_Z_MAX =  0.10,  1.10  # MIN low enough to cover the pick area (z~0.28)
    CLOUD_THROTTLE = 0.05      # 20 Hz processing — twice the reaction speed
    DETOUR_HEIGHT = 0.22
    DEBOUNCE_FRAMES = 2        # require N consecutive frames over threshold before injecting
    OBSTACLE_TTL = 0.8         # s — auto-remove sphere if not re-detected (faster clear when hand leaves)
    EEF_GUARD_RADIUS = 0.16    # m — ignore points within this of EEF (= the gripper extending past tool0)
    OBSTACLE_MOVE_EPS = 0.03   # m — sphere repositions on smaller hand movements (finer tracking)
    PREEMPT_DIST = 0.20        # m — if EEF gets this close to sphere mid-motion, cancel + replan
    WARN_DIST = 0.30           # m — log a "getting close" warning at this distance
    DECEL_WAIT = 2.0           # s — wait for arm to fully stop after a cancel before replanning
    MAX_REPLAN_DEPTH = 2       # cap recursive mid-execution replans
    # Robot self-filter is two-pass:
    #   - Line segments along the kinematic chain (covers the link shafts)
    #   - Sphere at each joint origin (covers the bulky joint housings)
    # Tuned tight: arm body covered, but small enough that a hand at typical
    # workspace distance from the arm (>15 cm from any joint) still survives.
    ARM_SEG_RADIUS   = 0.12    # m — tube around link shafts
    ARM_JOINT_RADIUS = 0.15    # m — sphere at each joint for housing bulge
    ARM_LINK_PAIRS = (
        ('base_link',      'shoulder_link'),
        ('shoulder_link',  'upper_arm_link'),
        ('upper_arm_link', 'forearm_link'),
        ('forearm_link',   'wrist_1_link'),
        ('wrist_1_link',   'wrist_2_link'),
        ('wrist_2_link',   'wrist_3_link'),
        ('wrist_3_link',   'tool0'),
    )

    def _cloud_cb(self, msg: PointCloud2):
        """Detect foreign obstacles in the workspace.

        Pipeline:
          1. depth filter in camera frame
          2. batch TF → base_link (preserve RGB alignment)
          3. workspace bounding-box filter
          4. color filter — drop robot-colored (gray/white/black) points
          5. robot self-filter — drop points geometrically inside the arm
          6. count remaining points; compute centroid when triggered
        """
        self._camera_ok = True

        now = time.time()
        if now - self._last_cloud_t < self.CLOUD_THROTTLE:
            return
        self._last_cloud_t = now

        # Try to read RGB along with XYZ. RealSense /depth/color/points has it.
        rgb = None
        try:
            pts_full = point_cloud2.read_points_numpy(
                msg, field_names=('x', 'y', 'z', 'rgb'), skip_nans=True)
            if pts_full is not None and len(pts_full) > 0 and pts_full.dtype.names:
                pts = np.column_stack([pts_full['x'], pts_full['y'], pts_full['z']])
                # rgb is float32; bit-cast to uint32 to extract bytes
                rgb_u32 = pts_full['rgb'].astype(np.float32).view(np.uint32)
                rgb = np.column_stack([
                    ((rgb_u32 >> 16) & 0xFF).astype(np.int16),  # R
                    ((rgb_u32 >> 8)  & 0xFF).astype(np.int16),  # G
                    ( rgb_u32        & 0xFF).astype(np.int16),  # B
                ])
            else:
                pts = None
        except Exception:
            pts = None

        # Fallback: xyz only if rgb read failed
        if pts is None:
            try:
                raw = point_cloud2.read_points_numpy(
                    msg, field_names=('x', 'y', 'z'), skip_nans=True)
            except Exception:
                return
            if raw is None or len(raw) == 0:
                return
            if raw.ndim == 1 and raw.dtype.names:
                pts = np.stack([raw['x'], raw['y'], raw['z']], axis=-1)
            else:
                pts = np.asarray(raw).reshape(-1, 3)
            rgb = None

        if len(pts) == 0:
            return

        # 1) depth filter (camera frame, cheap)
        z_cam = pts[:, 2]
        depth_mask = (z_cam > self.DEPTH_MIN) & (z_cam < self.DEPTH_MAX)
        pts = pts[depth_mask]
        if rgb is not None:
            rgb = rgb[depth_mask]
        if len(pts) == 0:
            return

        # 2) batch TF → base_link
        try:
            tf = self.tf_buffer.lookup_transform(
                'base_link', msg.header.frame_id, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05))
        except Exception:
            return
        R = self._quat_to_matrix(tf.transform.rotation)
        t = np.array([
            tf.transform.translation.x,
            tf.transform.translation.y,
            tf.transform.translation.z,
        ])
        pts_base = pts @ R.T + t

        # 3) workspace box filter
        ws_mask = (
            (pts_base[:, 0] >= self.WS_X_MIN) & (pts_base[:, 0] <= self.WS_X_MAX) &
            (pts_base[:, 1] >= self.WS_Y_MIN) & (pts_base[:, 1] <= self.WS_Y_MAX) &
            (pts_base[:, 2] >= self.WS_Z_MIN) & (pts_base[:, 2] <= self.WS_Z_MAX)
        )
        ws_pts = pts_base[ws_mask]
        ws_rgb = rgb[ws_mask] if rgb is not None else None

        # 4) color filter — drop robot-colored points (gray/white/black)
        if self.USE_COLOR_FILTER and ws_rgb is not None and len(ws_pts) > 0:
            color_keep = ~self._is_robot_color(ws_rgb)
            ws_pts = ws_pts[color_keep]

        # 5) robot self-filter (geometric)
        if len(ws_pts) > 0:
            ws_pts = self._filter_robot_self(ws_pts)

        count = len(ws_pts)

        # Update baseline whenever the arm is parked
        if not self._executing:
            self._baseline_samples.append(count)
            if len(self._baseline_samples) > 20:
                self._baseline_samples.pop(0)
            if len(self._baseline_samples) >= 5:
                self._baseline_count = int(np.median(self._baseline_samples))
            if self._obstacle_present:
                self._remove_obstacle()
            self._obstacle_streak = 0
            return

        if self._baseline_count is None:
            return

        diff = count - self._baseline_count

        # TTL: clear stale sphere if no recent detection (hand moved away)
        if self._obstacle_present and (now - self._last_obstacle_seen) > self.OBSTACLE_TTL:
            self._remove_obstacle()
            self._obstacle_streak = 0
            return

        if diff > self.OBSTACLE_THRESHOLD and count > 0:
            self._obstacle_streak += 1
            if self._obstacle_streak < self.DEBOUNCE_FRAMES:
                return

            # Centroid of foreign-in-workspace points (already in base_link)
            cx, cy, cz = ws_pts.mean(axis=0)
            if self._inject_obstacle_at_xyz(float(cx), float(cy), float(cz), diff):
                self._last_obstacle_seen = now
                self.obstacle_detected = True
        else:
            self._obstacle_streak = max(0, self._obstacle_streak - 1)
            if self._obstacle_present and diff < self.OBSTACLE_THRESHOLD // 2:
                self._remove_obstacle()

    def _inject_obstacle_at_xyz(self, x: float, y: float, z: float, diff: int) -> bool:
        """Publish a collision sphere at base_link (x,y,z). Caller pre-filtered.

        Returns True when an obstacle is registered (newly published OR refreshed
        existing), False if dedup decided no scene update was needed.
        """
        # Dedupe: barely moved → keep existing sphere, just refresh TTL
        if self._last_obstacle_xyz is not None:
            lx, ly, lz = self._last_obstacle_xyz
            if (x - lx) ** 2 + (y - ly) ** 2 + (z - lz) ** 2 < self.OBSTACLE_MOVE_EPS ** 2:
                return self._obstacle_present

        if not self._obstacle_present:
            self.get_logger().warn(
                f'Obstacle detected ({diff} pts) at base_link '
                f'({x:.2f}, {y:.2f}, {z:.2f})')

        obj = CollisionObject()
        obj.header.frame_id = 'base_link'
        obj.header.stamp = self.get_clock().now().to_msg()
        obj.id = 'detected_obstacle'
        obj.operation = CollisionObject.ADD
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.SPHERE
        prim.dimensions = [float(self.SPHERE_RADIUS)]
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = float(x), float(y), float(z)
        pose.orientation.w = 1.0
        obj.primitives = [prim]
        obj.primitive_poses = [pose]

        scene = PlanningSceneMsg()
        scene.is_diff = True
        scene.world.collision_objects = [obj]
        req = ApplyPlanningScene.Request()
        req.scene = scene
        fut = self.apply_scene_client.call_async(req)
        self._spin(fut, timeout_sec=0.15)
        self._obstacle_present = True
        self._last_obstacle_xyz = (x, y, z)
        return True

    def _is_robot_color(self, rgb: np.ndarray) -> np.ndarray:
        """Boolean mask: True for points whose color matches the robot.

        UR12e palette: light gray housing, black accents, light-blue/teal trim.
        Detection is order-tolerant where possible:
          - GRAY: low channel spread, mid-to-high brightness (any neutral tone)
          - BLACK: very low total brightness
          - COOL: red noticeably lower than max(green, blue) — catches all
            blues / teals / cyans / light-blues regardless of exact shade
        SKIN protection re-includes pixels that look like human skin so a hand
        right next to the arm still survives the filter.

        rgb: (N, 3) int16 array of (R, G, B) in 0..255.
        """
        r = rgb[:, 0]
        g = rgb[:, 1]
        b = rgb[:, 2]
        s = r + g + b
        max_diff = np.maximum(
            np.abs(r - g),
            np.maximum(np.abs(g - b), np.abs(r - b)),
        )

        # Gray covers white, off-white, mid-gray plastic — anything roughly neutral
        is_gray  = (s > 320) & (max_diff < 45)
        is_black = s < self.COLOR_BLACK_SUM
        # COOL — red is the LOW channel, max(g,b) bright. This single rule covers
        # light blue, teal, cyan, blue-gray, regardless of whether B or G is the
        # dominant channel, so it survives any RGB-vs-BGR byte-order ambiguity.
        is_cool = (np.maximum(g, b) > r + 20) & (np.maximum(g, b) > 100)

        # Skin protection (Kovac et al. RGB rule, relaxed):
        is_skin = (
            (r > 95) & (g > 40) & (b > 20) &
            (r > g) & (r > b) &
            (np.abs(r - g) > 15) &
            (max_diff > 15)
        )

        return (is_gray | is_black | is_cool) & ~is_skin

    def _filter_robot_self(self, ws_pts: np.ndarray) -> np.ndarray:
        """Drop points lying on the arm OR on the swept volume the arm just moved through.

        Two-stage:
          1. Look up current TF positions of every link
          2. Apply the geometric filter against CURRENT positions AND the last
             few historical snapshots, so transient points revealed by arm
             motion don't register as phantom obstacles before the baseline
             catches up.
        """
        if len(ws_pts) == 0:
            return ws_pts

        # Look up all unique link positions for the current frame
        link_pos = {}
        for from_link, to_link in self.ARM_LINK_PAIRS:
            for name in (from_link, to_link):
                if name in link_pos:
                    continue
                try:
                    tf = self.tf_buffer.lookup_transform(
                        'base_link', name, rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=0.02))
                    link_pos[name] = np.array([
                        tf.transform.translation.x,
                        tf.transform.translation.y,
                        tf.transform.translation.z,
                    ])
                except Exception:
                    pass

        if not link_pos:
            return ws_pts

        # Append current positions to history, then filter against ALL recent
        # snapshots — that's the swept-volume dilation that kills the
        # "newly-revealed background" phantoms. Snapshot to a list because the
        # callback group is reentrant (a second cloud_cb mutating the deque
        # during this iteration would raise).
        self._arm_pos_history.append(link_pos)
        history_snapshot = list(self._arm_pos_history)

        keep = np.ones(len(ws_pts), dtype=bool)
        for snap in history_snapshot:
            keep &= self._self_filter_mask(ws_pts, snap)

        return ws_pts[keep]

    def _self_filter_mask(self, ws_pts: np.ndarray, link_pos: dict) -> np.ndarray:
        """Return boolean keep-mask for ws_pts against ONE arm pose snapshot."""
        keep = np.ones(len(ws_pts), dtype=bool)

        # Pass 1: line-segment exclusion along link shafts
        seg_r2 = self.ARM_SEG_RADIUS ** 2
        for from_link, to_link in self.ARM_LINK_PAIRS:
            if from_link not in link_pos or to_link not in link_pos:
                continue
            a = link_pos[from_link]
            b = link_pos[to_link]
            ab = b - a
            ab2 = float(ab @ ab) + 1e-9
            ap = ws_pts - a
            t_param = np.clip((ap @ ab) / ab2, 0.0, 1.0)
            proj = a + t_param[:, None] * ab
            diff = ws_pts - proj
            d2 = (diff * diff).sum(axis=1)
            keep &= (d2 > seg_r2)

        # Pass 2: sphere at each joint origin (covers housing bulges)
        joint_r2 = self.ARM_JOINT_RADIUS ** 2
        for pos in link_pos.values():
            diff = ws_pts - pos
            d2 = (diff * diff).sum(axis=1)
            keep &= (d2 > joint_r2)

        # Pass 3: extra guard around tool0 for the gripper extending past the chain
        if 'tool0' in link_pos:
            diff = ws_pts - link_pos['tool0']
            d2 = (diff * diff).sum(axis=1)
            keep &= (d2 > self.EEF_GUARD_RADIUS ** 2)

        return keep

    def _wait_arm_stopped(self, timeout: float = 3.0) -> bool:
        """Block until joint velocities settle to ~zero (arm physically stopped).

        Cancelling a goal in MoveIt only tells the action server to stop
        publishing setpoints — the UR controller still needs time to decelerate.
        Sending a fresh trajectory before that finishes triggers CONTROL_FAILED
        and a protective stop. Waiting for joint velocities to drop solves it.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            js = self.current_joint_state
            if js is not None and js.velocity:
                if max(abs(v) for v in js.velocity) < 0.01:
                    return True
            time.sleep(0.05)
        self.get_logger().warn('Arm did not settle within timeout — proceeding anyway')
        return False

    def _path_midpoint(self):
        """Fallback: midpoint of home → current goal in base_link."""
        if self._current_goal_pose is None or self.home_eef_pose is None:
            return None, None, None
        h, g = self.home_eef_pose.position, self._current_goal_pose.position
        return (h.x + g.x) / 2, (h.y + g.y) / 2, (h.z + g.z) / 2

    def _remove_obstacle(self):
        obj = CollisionObject()
        obj.header.frame_id = 'base_link'
        obj.header.stamp = self.get_clock().now().to_msg()
        obj.id = 'detected_obstacle'
        obj.operation = CollisionObject.REMOVE
        scene = PlanningSceneMsg()
        scene.is_diff = True
        scene.world.collision_objects = [obj]
        req = ApplyPlanningScene.Request()
        req.scene = scene
        fut = self.apply_scene_client.call_async(req)
        self._spin(fut, timeout_sec=0.15)
        self._obstacle_present = False

    def _current_joints(self) -> List[float]:
        if self.current_joint_state is None:
            return [0.0] * 6
        out = []
        for name in self.joint_names:
            try:
                idx = list(self.current_joint_state.name).index(name)
                out.append(self.current_joint_state.position[idx])
            except (ValueError, IndexError):
                out.append(0.0)
        return out

    # Home: 180 -90 90 -90 -90 90 deg
    HOME_JOINTS = [3.14159, -1.5708, 1.5708, -1.5708, -1.5708, 1.5708]

    def _random_seed(self) -> List[float]:
        limits = [(-3.14, 3.14), (-2.5, 0.0), (-2.0, 2.0),
                  (-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14)]
        return [np.random.uniform(lo, hi) for lo, hi in limits]

    def _manipulability(self, joints: List[float]) -> float:
        score = 1.0
        for j in joints:
            score *= min(abs(j - 3.14), abs(j + 3.14)) / 3.14
        return score

    def compute_home_fk(self) -> Optional[Pose]:
        """FK at home joints — gives us the reference EEF position and orientation."""
        req = GetPositionFK.Request()
        req.header.frame_id = 'base_link'
        req.fk_link_names = ['tool0']
        req.robot_state.joint_state.name = self.joint_names
        req.robot_state.joint_state.position = self.HOME_JOINTS

        fut = self.fk_client.call_async(req)
        self._spin(fut, timeout_sec=5.0)

        if fut.result() and fut.result().error_code.val == MoveItErrorCodes.SUCCESS:
            pose = fut.result().pose_stamped[0].pose
            self.get_logger().info(
                f'Home EEF pos=({pose.position.x:.3f}, {pose.position.y:.3f},'
                f' {pose.position.z:.3f})  '
                f'ori=({pose.orientation.x:.3f}, {pose.orientation.y:.3f},'
                f' {pose.orientation.z:.3f}, {pose.orientation.w:.3f})')
            return pose

        self.get_logger().error('FK at home failed — is MoveIt running?')
        return None

    def _build_poses(self):
        """Pick = down from home, Place = right of home. Orientation fixed to home EEF."""
        h = self.home_eef_pose
        ox, oy, oz, ow = h.orientation.x, h.orientation.y, h.orientation.z, h.orientation.w

        self.pick_pose = self._make_pose(
            h.position.x,
            h.position.y,
            h.position.z + self.get_parameter('pick_z_offset').value,
            ox, oy, oz, ow,
        )
        self.place_pose = self._make_pose(
            h.position.x,
            h.position.y + self.get_parameter('place_y_offset').value,
            h.position.z,
            ox, oy, oz, ow,
        )

        self.get_logger().info(
            f'Pick:  ({self.pick_pose.position.x:.3f},'
            f' {self.pick_pose.position.y:.3f},'
            f' {self.pick_pose.position.z:.3f})')
        self.get_logger().info(
            f'Place: ({self.place_pose.position.x:.3f},'
            f' {self.place_pose.position.y:.3f},'
            f' {self.place_pose.position.z:.3f})')

    # ── IK ────────────────────────────────────────────────────────────────────

    def compute_ik_solutions(self, target: Pose, num: int = 30) -> List[IKSolution]:
        solutions, seen = [], set()
        self.get_logger().info(f'Computing {num} IK solutions...')

        for i in range(num):
            if i == 0:
                seed = self._current_joints()
            elif i == 1:
                seed = self.HOME_JOINTS
            else:
                seed = self._random_seed()

            req = GetPositionIK.Request()
            req.ik_request.group_name = 'ur_manipulator'
            req.ik_request.robot_state.joint_state.name = self.joint_names
            req.ik_request.robot_state.joint_state.position = seed
            req.ik_request.pose_stamped.header.frame_id = 'base_link'
            req.ik_request.pose_stamped.header.stamp = self.get_clock().now().to_msg()
            req.ik_request.pose_stamped.pose = target
            req.ik_request.timeout.nanosec = 50_000_000  # 50 ms per attempt
            # avoid_collisions=False so IK gives us all kinematic solutions; OMPL
            # in plan_to_joints handles the actual collision avoidance during planning.
            req.ik_request.avoid_collisions = False

            fut = self.ik_client.call_async(req)
            self._spin(fut, timeout_sec=0.3)

            if fut.result() and fut.result().error_code.val == MoveItErrorCodes.SUCCESS:
                pos = []
                for name in self.joint_names:
                    try:
                        idx = list(fut.result().solution.joint_state.name).index(name)
                        pos.append(fut.result().solution.joint_state.position[idx])
                    except Exception:
                        break
                if len(pos) == 6:
                    key = tuple(round(p, 1) for p in pos)
                    if key not in seen:
                        seen.add(key)
                        solutions.append(IKSolution(pos, self._manipulability(pos), True))

        # Drop only deeply singular configs (manip ~0); keep moderately-poor ones.
        solutions = [s for s in solutions if s.manipulability > 0.005]
        # Sort by joint-space distance from current state (closest first). High
        # manipulability ≠ short path — the closest config is what produces the
        # shortest motion, so try those first instead of wading through 16 wrist
        # flips before finding the obvious one.
        cur = self._current_joints()
        def _joint_dist(s: IKSolution) -> float:
            return sum((a - b) ** 2 for a, b in zip(s.joint_positions, cur)) ** 0.5
        solutions.sort(key=_joint_dist)
        self.get_logger().info(f'Found {len(solutions)} usable IK solutions')
        return solutions

    # ── planning / execution ──────────────────────────────────────────────────

    def _check_cartesian_blocked(self, target: Pose) -> bool:
        """Returns True if path to target is blocked. Logs fraction for diagnostics."""
        req = GetCartesianPath.Request()
        req.header.frame_id = 'base_link'
        req.header.stamp = self.get_clock().now().to_msg()
        req.group_name = 'ur_manipulator'
        req.link_name = 'tool0'
        req.waypoints = [target]
        req.max_step = 0.02
        req.jump_threshold = 0.0
        req.avoid_collisions = True
        req.max_velocity_scaling_factor     = self.get_parameter('vel_scale').value
        req.max_acceleration_scaling_factor = self.get_parameter('acc_scale').value
        if self.current_joint_state is not None:
            req.start_state.joint_state = self.current_joint_state

        fut = self.cartesian_client.call_async(req)
        self._spin(fut, timeout_sec=5.0)

        if not fut.result():
            self.get_logger().info('Obstacle check: no response from planner')
            return True
        frac = fut.result().fraction
        self.get_logger().info(f'Obstacle check: path fraction = {frac:.2f}')
        return frac < 0.99

    def plan_cartesian(self, target: Pose) -> Optional[JointTrajectory]:
        """Straight-line Cartesian path from current pose to target. Returns None if blocked."""
        req = GetCartesianPath.Request()
        req.header.frame_id = 'base_link'
        req.header.stamp = self.get_clock().now().to_msg()
        req.group_name = 'ur_manipulator'
        req.link_name = 'tool0'
        req.waypoints = [target]
        req.max_step = 0.01          # 1 cm resolution
        req.jump_threshold = 0.0     # disabled
        req.avoid_collisions = True
        req.max_velocity_scaling_factor     = self.get_parameter('vel_scale').value
        req.max_acceleration_scaling_factor = self.get_parameter('acc_scale').value

        # Seed from current joint state
        if self.current_joint_state is not None:
            req.start_state.joint_state = self.current_joint_state

        fut = self.cartesian_client.call_async(req)
        self._spin(fut, timeout_sec=10.0)

        if not fut.result():
            return None
        # fraction == 1.0 means the full path was computed collision-free
        if fut.result().fraction < 0.99:
            self.get_logger().info(
                f'Cartesian path only {fut.result().fraction * 100:.0f}% complete — blocked')
            return None
        return fut.result().solution.joint_trajectory

    def _current_eef_pose(self) -> Optional[Pose]:
        req = GetPositionFK.Request()
        req.header.frame_id = 'base_link'
        req.fk_link_names = ['tool0']
        if self.current_joint_state is not None:
            req.robot_state.joint_state = self.current_joint_state
        fut = self.fk_client.call_async(req)
        self._spin(fut, timeout_sec=2.0)
        if fut.result() and fut.result().error_code.val == MoveItErrorCodes.SUCCESS:
            return fut.result().pose_stamped[0].pose
        return None

    def _try_cartesian_waypoints(self, waypoints: List[Pose]) -> Optional[JointTrajectory]:
        req = GetCartesianPath.Request()
        req.header.frame_id = 'base_link'
        req.header.stamp = self.get_clock().now().to_msg()
        req.group_name = 'ur_manipulator'
        req.link_name = 'tool0'
        req.waypoints = waypoints
        req.max_step = 0.02
        req.jump_threshold = 0.0
        req.avoid_collisions = True
        req.max_velocity_scaling_factor     = self.get_parameter('vel_scale').value
        req.max_acceleration_scaling_factor = self.get_parameter('acc_scale').value
        if self.current_joint_state is not None:
            req.start_state.joint_state = self.current_joint_state

        fut = self.cartesian_client.call_async(req)
        self._spin(fut, timeout_sec=6.0)
        if not fut.result():
            return None
        if fut.result().fraction < 0.95:
            return None
        return fut.result().solution.joint_trajectory

    def _wp(self, x, y, z, ori) -> Pose:
        p = Pose()
        p.position.x, p.position.y, p.position.z = float(x), float(y), float(z)
        p.orientation = ori
        return p

    def plan_arc_detour(self, target: Pose) -> Optional[JointTrajectory]:
        """Cartesian detour around an obstacle on the direct line.

        Picks a detour pattern based on motion direction:
          - Mostly horizontal (PLACE): lift UP, go OVER, come DOWN
          - Mostly vertical (PICK): step to the SIDE, descend at offset, slide IN
        Tries multiple side offsets if the first is also blocked.
        """
        current = self._current_eef_pose() or self.home_eef_pose
        if current is None:
            return None

        c = current.position
        t = target.position
        ori = target.orientation
        dx, dy, dz = t.x - c.x, t.y - c.y, t.z - c.z
        horiz = (dx * dx + dy * dy) ** 0.5

        if horiz > 0.05:
            # Horizontal-dominant: lift over the obstacle
            lift_z = max(c.z, t.z) + self.DETOUR_HEIGHT
            wps = [
                self._wp(c.x, c.y, lift_z, ori),
                self._wp(t.x, t.y, lift_z, ori),
                target,
            ]
            traj = self._try_cartesian_waypoints(wps)
            if traj is not None:
                self.get_logger().info(f'Arc detour (lift {lift_z:.2f}) planned')
                return traj
            self.get_logger().info('Lift detour blocked')
            return None

        # Vertical-dominant (PICK): try side detours in Y, then X
        for axis, sign in [('y', +1), ('y', -1), ('x', +1), ('x', -1)]:
            off = 0.18 * sign
            ox, oy = (off, 0) if axis == 'x' else (0, off)
            wps = [
                self._wp(c.x + ox, c.y + oy, c.z, ori),
                self._wp(t.x + ox, t.y + oy, t.z, ori),
                target,
            ]
            traj = self._try_cartesian_waypoints(wps)
            if traj is not None:
                self.get_logger().info(
                    f'Side detour (offset {axis}{off:+.2f}) planned')
                return traj

        self.get_logger().info('All side detours blocked')
        return None

    def plan_to_joints(self, target: List[float]) -> Optional[JointTrajectory]:
        """Joint-space plan to a specific joint configuration (used for IK-based replanning).

        Uses plan_only=True so we get a trajectory back without auto-executing —
        execute_trajectory() is what actually drives the controller.
        """
        goal = MoveGroup.Goal()
        goal.request.group_name = 'ur_manipulator'
        goal.request.pipeline_id = 'ompl'
        goal.request.planner_id = 'RRTConnectkConfigDefault'
        goal.request.num_planning_attempts = 1
        goal.request.allowed_planning_time = 0.5
        goal.request.max_velocity_scaling_factor     = self.get_parameter('vel_scale').value
        goal.request.max_acceleration_scaling_factor = self.get_parameter('acc_scale').value
        goal.planning_options.plan_only = True
        goal.planning_options.replan = False

        c = Constraints()
        for name, pos in zip(self.joint_names, target):
            jc = JointConstraint()
            jc.joint_name, jc.position = name, pos
            jc.tolerance_above = jc.tolerance_below = 0.01
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        goal.request.goal_constraints.append(c)

        fut = self.move_client.send_goal_async(goal)
        self._spin(fut, timeout_sec=2.0)
        if not fut.done() or fut.result() is None or not fut.result().accepted:
            return None

        res_fut = fut.result().get_result_async()
        self._spin(res_fut, timeout_sec=3.0)
        res = res_fut.result()
        if res is None or res.result is None:
            return None
        if res.result.error_code.val == MoveItErrorCodes.SUCCESS:
            return res.result.planned_trajectory.joint_trajectory
        return None

    def execute_trajectory(self, traj: JointTrajectory, goal_pose: Pose = None,
                           depth: int = 0) -> bool:
        """Send a pre-planned trajectory and actively monitor for preempt.

        Same path-intersection + EEF-proximity logic as move_to_pose. If we
        preempt mid-execution, recurse through move_to_pose so the full primary
        + START_STATE retry path fires for the replan.
        """
        if depth > self.MAX_REPLAN_DEPTH:
            self.get_logger().warn('execute_trajectory: max replan depth reached')
            return False

        self._current_goal_pose = goal_pose
        goal = ExecuteTrajectory.Goal()
        goal.trajectory.joint_trajectory = traj
        self.obstacle_detected = False
        self._executing = True

        fut = self.exec_client.send_goal_async(goal)
        self._spin(fut, timeout_sec=5.0)
        if not fut.done() or fut.result() is None or not fut.result().accepted:
            self.get_logger().error(
                'Controller rejected trajectory — likely PROTECTIVE STOP. '
                'Reset on teach pendant.')
            self._protective_stop = True
            self._executing = False
            self._current_goal_pose = None
            return False

        goal_handle = fut.result()
        res_fut = goal_handle.get_result_async()

        # Active monitoring (same logic as move_to_pose)
        plan_obstacle_xyz = self._last_obstacle_xyz if self._obstacle_present else None
        last_logged_xyz = self._last_obstacle_xyz
        last_warn_t = 0.0
        preempted = False
        preempt_reason = ''
        deadline = time.time() + 60.0

        while not res_fut.done() and time.time() < deadline:
            time.sleep(0.1)

            if not self._obstacle_present or self._last_obstacle_xyz is None:
                continue
            sx, sy, sz = self._last_obstacle_xyz

            if last_logged_xyz != self._last_obstacle_xyz:
                self.get_logger().info(
                    f'EXEC: obstacle update → ({sx:.2f}, {sy:.2f}, {sz:.2f})')
                last_logged_xyz = self._last_obstacle_xyz

            # Gate: only consider preempt if sphere moved since plan time
            sphere_changed = True
            if plan_obstacle_xyz is not None:
                px, py, pz = plan_obstacle_xyz
                moved2 = (sx - px) ** 2 + (sy - py) ** 2 + (sz - pz) ** 2
                if moved2 < self.OBSTACLE_MOVE_EPS ** 2:
                    sphere_changed = False

            eef = self._eef_xyz_from_tf()

            # Path-intersection check (only if sphere is new/moved)
            if sphere_changed and self._planned_eef_xyz and eef is not None:
                ex, ey, ez = eef
                best_i, best_d = 0, float('inf')
                for i, (wx, wy, wz) in enumerate(self._planned_eef_xyz):
                    d2 = (wx - ex) ** 2 + (wy - ey) ** 2 + (wz - ez) ** 2
                    if d2 < best_d:
                        best_d, best_i = d2, i
                threshold2 = (self.PATH_CLEARANCE + self.SPHERE_RADIUS) ** 2
                for i in range(best_i, len(self._planned_eef_xyz)):
                    wx, wy, wz = self._planned_eef_xyz[i]
                    if (sx - wx) ** 2 + (sy - wy) ** 2 + (sz - wz) ** 2 < threshold2:
                        preempted = True
                        preempt_reason = (f'path waypoint {i}/'
                                          f'{len(self._planned_eef_xyz)} within clearance')
                        break

            if not preempted and eef is not None:
                ex, ey, ez = eef
                dist = ((sx - ex) ** 2 + (sy - ey) ** 2 + (sz - ez) ** 2) ** 0.5
                if dist < self.PREEMPT_DIST:
                    preempted = True
                    preempt_reason = f'EEF {dist*100:.0f}cm from obstacle'
                elif dist < self.WARN_DIST and time.time() - last_warn_t > 1.0:
                    last_warn_t = time.time()
                    self.get_logger().info(
                        f'EXEC: EEF {dist*100:.0f}cm from obstacle (closing in)')

            if preempted:
                self.get_logger().warn(f'EXEC: PREEMPTING — {preempt_reason}')
                cancel_fut = goal_handle.cancel_goal_async()
                self._spin(cancel_fut, timeout_sec=1.5)
                self._wait_arm_stopped(timeout=self.DECEL_WAIT)
                time.sleep(0.12)  # short buffer for MoveIt exec manager to clean up
                break

        self._executing = False
        self._current_goal_pose = None

        if preempted and goal_pose is not None:
            self.get_logger().info(
                f'EXEC: replanning to original goal (depth {depth + 1})')
            return self.move_to_pose(goal_pose, 'REPLAN', depth=depth + 1)

        result = res_fut.result()
        if result is None or result.result is None:
            return False
        return result.result.error_code.val == MoveItErrorCodes.SUCCESS

    def gripper(self, open: bool):
        goal = GripperCommand.Goal()
        goal.command.position   = 0.025 if open else 0.0
        goal.command.max_effort = 50.0

        fut = self.gripper_client.send_goal_async(goal)
        self._spin(fut, timeout_sec=5.0)
        if not fut.done() or not fut.result() or not fut.result().accepted:
            self.get_logger().warn('Gripper goal rejected')
            return

        res_fut = fut.result().get_result_async()
        self._spin(res_fut, timeout_sec=6.0)  # gripper node polls up to 5 s internally

    # ── core replanning logic ─────────────────────────────────────────────────

    def _eef_xyz_from_tf(self):
        """Read tool0 → base_link from TF (cheap, no FK service call)."""
        try:
            t = self.tf_buffer.lookup_transform(
                'base_link', 'tool0', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05))
            return (t.transform.translation.x,
                    t.transform.translation.y,
                    t.transform.translation.z)
        except Exception:
            return None

    @staticmethod
    def _quat_to_matrix(q) -> np.ndarray:
        """Quaternion (with .x .y .z .w) → 3x3 rotation matrix."""
        xx, yy, zz = q.x * q.x, q.y * q.y, q.z * q.z
        xy, xz, yz = q.x * q.y, q.x * q.z, q.y * q.z
        wx, wy, wz = q.w * q.x, q.w * q.y, q.w * q.z
        return np.array([
            [1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy)    ],
            [2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx)    ],
            [2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy)],
        ])

    def move_to_pose(self, target: Pose, label: str, depth: int = 0) -> bool:
        """Plan + execute via MoveGroup action with pose constraints.

        Uses OMPL (joint-space) so the planner can route around any collision
        objects in the scene. planning_options.replan=True makes the action
        server re-plan if the scene changes mid-execution.

        Active mid-execution monitoring: if the EEF gets within PREEMPT_DIST of
        a sphere, cancel the goal cleanly, wait for the arm to decelerate, then
        recursively replan from the new state. Capped at MAX_REPLAN_DEPTH.
        """
        if depth > self.MAX_REPLAN_DEPTH:
            self.get_logger().warn(f'{label}: max replan depth reached — giving up')
            return False
        goal = MoveGroup.Goal()
        goal.request.group_name = 'ur_manipulator'
        goal.request.pipeline_id = 'ompl'
        # BITstar is asymptotically optimal — given enough time, it produces
        # significantly shorter and smoother paths than RRTConnect. The longer
        # allowed_planning_time pays for itself in motion quality.
        goal.request.planner_id = 'BITstarkConfigDefault'
        goal.request.num_planning_attempts = 4
        goal.request.allowed_planning_time = 2.5
        goal.request.max_velocity_scaling_factor     = self.get_parameter('vel_scale').value
        goal.request.max_acceleration_scaling_factor = self.get_parameter('acc_scale').value

        c = Constraints()

        # Position: tool0 must reach target.position within 5 mm
        pc = PositionConstraint()
        pc.header.frame_id = 'base_link'
        pc.link_name = 'tool0'
        region = SolidPrimitive()
        region.type = SolidPrimitive.SPHERE
        region.dimensions = [0.005]
        bv = BoundingVolume()
        bv.primitives = [region]
        ref_pose = Pose()
        ref_pose.position = target.position
        ref_pose.orientation.w = 1.0
        bv.primitive_poses = [ref_pose]
        pc.constraint_region = bv
        pc.weight = 1.0

        # Orientation: keep the EEF roughly aligned with home
        oc = OrientationConstraint()
        oc.header.frame_id = 'base_link'
        oc.link_name = 'tool0'
        oc.orientation = target.orientation
        oc.absolute_x_axis_tolerance = 0.1
        oc.absolute_y_axis_tolerance = 0.1
        oc.absolute_z_axis_tolerance = 0.1
        oc.weight = 1.0

        c.position_constraints = [pc]
        c.orientation_constraints = [oc]
        goal.request.goal_constraints.append(c)

        # Auto-replan when planning scene changes (i.e. when our sphere is added/moved)
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 5
        goal.planning_options.replan_delay = 0.2

        self.obstacle_detected = False
        self._executing = True
        self._current_goal_pose = target

        fut = self.move_client.send_goal_async(goal)
        self._spin(fut, timeout_sec=5.0)
        if not fut.done() or fut.result() is None or not fut.result().accepted:
            self.get_logger().error(f'{label}: MoveGroup goal rejected')
            self._executing = False
            self._current_goal_pose = None
            return False

        goal_handle = fut.result()
        res_fut = goal_handle.get_result_async()

        # Active monitoring (every 100 ms during execution):
        #   1. PATH INTERSECTION — does the sphere sit on any UPCOMING planned waypoint?
        #      Fires the moment the hand enters the path corridor, well before
        #      the arm gets near it. This is what makes mid-exec replan feel real-time.
        #   2. EEF PROXIMITY — last-resort safety net if (1) misses (e.g. no plan cached).
        # Snapshot the obstacle state at plan time. If the sphere doesn't move
        # from this position, the current plan already accounts for it (OMPL
        # routed around it during planning). Only preempt when the sphere has
        # MOVED since planning — that's the actual signal of a new threat.
        plan_obstacle_xyz = self._last_obstacle_xyz if self._obstacle_present else None
        last_logged_xyz = self._last_obstacle_xyz
        last_warn_t = 0.0
        preempted = False
        preempt_reason = ''
        deadline = time.time() + 60.0

        while not res_fut.done() and time.time() < deadline:
            time.sleep(0.1)

            if not self._obstacle_present or self._last_obstacle_xyz is None:
                continue

            sx, sy, sz = self._last_obstacle_xyz

            # Log sphere movement for visibility
            if last_logged_xyz != self._last_obstacle_xyz:
                self.get_logger().info(
                    f'{label}: obstacle update → ({sx:.2f}, {sy:.2f}, {sz:.2f})')
                last_logged_xyz = self._last_obstacle_xyz

            # Gate: skip path-intersection check if sphere hasn't moved since
            # plan time. The plan already accounts for the original sphere.
            sphere_changed = True
            if plan_obstacle_xyz is not None:
                px, py, pz = plan_obstacle_xyz
                moved2 = (sx - px) ** 2 + (sy - py) ** 2 + (sz - pz) ** 2
                if moved2 < self.OBSTACLE_MOVE_EPS ** 2:
                    sphere_changed = False

            eef = self._eef_xyz_from_tf()

            # (1) Path-intersection check (only if sphere is new/moved)
            if sphere_changed and self._planned_eef_xyz and eef is not None:
                ex, ey, ez = eef
                # Find index of nearest cached waypoint to current EEF — assume we're past it
                best_i, best_d = 0, float('inf')
                for i, (wx, wy, wz) in enumerate(self._planned_eef_xyz):
                    d2 = (wx - ex) ** 2 + (wy - ey) ** 2 + (wz - ez) ** 2
                    if d2 < best_d:
                        best_d, best_i = d2, i
                # Check sphere distance against waypoints AHEAD of us
                threshold2 = (self.PATH_CLEARANCE + self.SPHERE_RADIUS) ** 2
                hit_idx = -1
                for i in range(best_i, len(self._planned_eef_xyz)):
                    wx, wy, wz = self._planned_eef_xyz[i]
                    if (sx - wx) ** 2 + (sy - wy) ** 2 + (sz - wz) ** 2 < threshold2:
                        hit_idx = i
                        break
                if hit_idx >= 0:
                    preempted = True
                    preempt_reason = (
                        f'path waypoint {hit_idx}/{len(self._planned_eef_xyz)} '
                        f'within clearance of obstacle')

            # (2) EEF-proximity safety net
            if not preempted and eef is not None:
                ex, ey, ez = eef
                dist = ((sx - ex) ** 2 + (sy - ey) ** 2 + (sz - ez) ** 2) ** 0.5
                if dist < self.PREEMPT_DIST:
                    preempted = True
                    preempt_reason = f'EEF {dist*100:.0f}cm from obstacle'
                elif dist < self.WARN_DIST and time.time() - last_warn_t > 1.0:
                    last_warn_t = time.time()
                    self.get_logger().info(
                        f'{label}: EEF {dist*100:.0f}cm from obstacle (closing in)')

            if preempted:
                self.get_logger().warn(
                    f'{label}: PREEMPTING — {preempt_reason}')
                cancel_fut = goal_handle.cancel_goal_async()
                self._spin(cancel_fut, timeout_sec=1.5)
                self._wait_arm_stopped(timeout=self.DECEL_WAIT)
                time.sleep(0.12)  # short buffer for MoveIt exec manager to clean up
                break

        self._executing = False
        self._current_goal_pose = None

        if preempted:
            self.get_logger().info(
                f'{label}: replanning from current state (depth {depth + 1})')
            return self.move_to_pose(target, label, depth=depth + 1)

        res = res_fut.result()
        if res is None or res.result is None:
            self.get_logger().error(f'{label}: MoveGroup timed out')
            return False
        code = res.result.error_code.val
        if code == MoveItErrorCodes.SUCCESS:
            return True
        reasons = {
            -1: 'PLANNING_FAILED',
            -2: 'INVALID_MOTION_PLAN',
            -3: 'MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE',
            -4: 'CONTROL_FAILED',
            -6: 'TIMED_OUT',
            -7: 'PREEMPTED',
            -10: 'START_STATE_IN_COLLISION',
            -11: 'START_STATE_VIOLATES_PATH_CONSTRAINTS',
            -12: 'GOAL_IN_COLLISION',
            -13: 'GOAL_VIOLATES_PATH_CONSTRAINTS',
            -14: 'GOAL_CONSTRAINTS_VIOLATED',
            -31: 'INVALID_MOTION_PLAN',
            99999: 'FAILURE',
        }
        self.last_failure_code = code
        if code == MoveItErrorCodes.CONTROL_FAILED:
            self._protective_stop = True
        self.get_logger().warn(
            f'{label}: MoveGroup failed ({reasons.get(code, code)})')
        return False

    def move_with_replanning(self, pose: Pose, label: str) -> bool:
        self.get_logger().info(f'Moving to {label}...')

        if self._protective_stop:
            self.get_logger().error(
                'Protective stop latched — reset on teach pendant and re-run.')
            return False

        # Brief settle: let any in-flight detection from the previous motion finish.
        if self.obstacle_detected:
            self.get_logger().info('Obstacle flag set at start — settling 1s')
            time.sleep(1.0)

        # Primary: MoveGroup with pose constraints + replan=True. OMPL routes
        # around any collision objects in the scene. If START_STATE_IN_COLLISION
        # fires, the sphere is on top of the arm — clear it and retry once.
        if self.move_to_pose(pose, label):
            return True
        if self._protective_stop:
            return False

        if self.last_failure_code == MoveItErrorCodes.START_STATE_IN_COLLISION:
            self.get_logger().warn('Start state in collision — clearing sphere and retrying')
            self._remove_obstacle()
            self._obstacle_streak = 0
            self._last_obstacle_xyz = None
            time.sleep(0.3)
            if self.move_to_pose(pose, label):
                return True
            if self._protective_stop:
                return False

        # Secondary: kinematic redundancy. Solutions are sorted by joint distance
        # from current state — closest configs first → shortest paths first.
        # Skip path viz during this phase (each plan triggers /display_planned_path,
        # which would fire 25 FK calls × N configs and choke the IK service queue).
        self.get_logger().info('MoveGroup failed — trying IK redundancy')
        self._skip_path_viz = True
        try:
            solutions = self.compute_ik_solutions(
                pose, self.get_parameter('ik_attempts').value)
            if not solutions:
                self.get_logger().error(f'No IK solutions for {label}')
            else:
                pool_size = self.get_parameter('ik_pool_size').value
                max_len   = self.get_parameter('max_traj_length').value
                short_eps = self.get_parameter('short_path_eps').value

                # Try N candidates, collect the ones under max_len, pick the
                # SHORTEST. Early-exit if a candidate beats short_eps (no point
                # planning more if we already have a short path). Solutions are
                # already sorted by joint distance from current state, so short
                # paths come early.
                candidates = []  # (length, traj, idx)
                for i, sol in enumerate(solutions[:pool_size], 1):
                    if self._protective_stop:
                        return False
                    traj = self.plan_to_joints(sol.joint_positions)
                    if traj is None:
                        continue
                    length = self._traj_length(traj)
                    if length > max_len:
                        self.get_logger().info(
                            f'  config {i}: rejected (length {length:.2f} > {max_len:.2f})')
                        continue
                    self.get_logger().info(
                        f'  config {i}: planned (length {length:.2f})')
                    candidates.append((length, traj, i))
                    if length < short_eps:
                        break  # short enough — stop searching

                if candidates:
                    candidates.sort(key=lambda c: c[0])
                    length, traj, idx = candidates[0]
                    self.get_logger().info(
                        f'  best of {len(candidates)}: config {idx} '
                        f'(length {length:.2f}) — executing')
                    self._skip_path_viz = False
                    self._publish_eef_path(traj)
                    if self.execute_trajectory(traj, goal_pose=pose):
                        return True
                    if self._protective_stop:
                        return False
        finally:
            self._skip_path_viz = False

        # Tertiary: arc detour (cartesian over the obstacle). Last resort because
        # straight-line cartesian rarely navigates around 3D obstacles.
        self.get_logger().info('IK redundancy exhausted — trying arc detour')
        traj = self.plan_arc_detour(pose)
        if traj is not None and self.execute_trajectory(traj, goal_pose=pose):
            self.get_logger().info('  → SUCCESS via arc detour')
            return True

        self.get_logger().error(f'{label}: all strategies exhausted')
        return False

    # ── demo ─────────────────────────────────────────────────────────────────

    def run_demo(self):
        self.get_logger().info('UR12e + Hand-E Reactive Replanning Demo starting...')
        self.get_logger().info('Inject obstacle during motion: '
            'ros2 run reactive_replanning_ur12e insert_obstacle '
            '--ros-args -p x:=0.3 -p y:=0.2 -p z:=0.4')

        while self.current_joint_state is None:
            self.get_logger().info('Waiting for joint states...')
            time.sleep(1.0)

        self.get_logger().info('Computing home EEF pose via FK...')
        self.home_eef_pose = self.compute_home_fk()
        if self.home_eef_pose is None:
            self.get_logger().error('Cannot proceed without home FK — is MoveIt up?')
            return
        self._build_poses()

        # Wait for camera, then build a baseline (arm at home, no hand)
        self.get_logger().info('Waiting for camera point cloud (T2: rs_launch.py)...')
        deadline = time.time() + 10.0
        while not self._camera_ok and time.time() < deadline:
            time.sleep(0.5)
        if not self._camera_ok:
            self.get_logger().error(
                'No camera data! Run: ros2 launch realsense2_camera rs_launch.py pointcloud.enable:=true')
            self.get_logger().warn('Continuing without obstacle detection.')
        else:
            self.get_logger().info('Capturing baseline point count (3s, keep workspace clear)...')
            time.sleep(3.0)
            if self._baseline_count is not None:
                self.get_logger().info(
                    f'Baseline: {self._baseline_count} workspace points. '
                    f'Threshold: +{self.OBSTACLE_THRESHOLD} = obstacle.')
            else:
                self.get_logger().warn('Baseline not captured — detection may be unreliable.')

        self.get_logger().info('Press ENTER to start...')
        input()

        for cycle in range(3):
            if self._protective_stop:
                self.get_logger().error(
                    'Protective stop active — aborting demo. '
                    'Reset the UR teach pendant before re-running.')
                break

            self.get_logger().info(f'Cycle {cycle + 1}/3')

            self.gripper(open=True)
            if self.move_with_replanning(self.pick_pose, 'PICK'):
                self.get_logger().info('At pick — closing gripper')
                self.gripper(open=False)
                time.sleep(0.5)

                if self.move_with_replanning(self.place_pose, 'PLACE'):
                    self.get_logger().info('At place — opening gripper')
                    self.gripper(open=True)
                    time.sleep(0.5)
            else:
                self.get_logger().warn('Failed to reach pick position')

            # Clear sphere + reset baseline so next cycle starts fresh
            if self._obstacle_present:
                self._remove_obstacle()
            self.obstacle_detected = False
            self._obstacle_streak = 0
            self._last_obstacle_xyz = None
            self._baseline_count = None
            self._baseline_samples.clear()
            time.sleep(2.0)  # let baseline rebuild while arm is parked

        self.get_logger().info('Demo complete.')


def main(args=None):
    rclpy.init(args=args)
    node = ReactiveReplannerUR12e()

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    node._executor = executor

    thread = threading.Thread(target=node.run_demo, daemon=True)
    thread.start()

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
