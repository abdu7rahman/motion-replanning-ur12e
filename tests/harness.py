"""Loads the real ReactiveReplannerUR12e and drives _cloud_cb without ROS.

Only two things are faked: the point-cloud reader and the TF buffer. Every
filtering, thresholding and debounce decision below them is the node's own
code, so what these tests measure is the shipped pipeline.
"""
import io, os, sys, types, contextlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import scene  # noqa: E402


def _stub_ros():
    def mod(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        if "." in name:                    # attach to parent so `pkg.sub` resolves
            parent, child = name.rsplit(".", 1)
            if parent in sys.modules:
                setattr(sys.modules[parent], child, m)
        return m

    class Any:
        def __init__(self, *a, **k): pass
        def __getattr__(self, n): return Any()
        def __call__(self, *a, **k): return Any()

    mod("rclpy", init=lambda *a, **k: None, spin=lambda *a, **k: None,
        shutdown=lambda *a, **k: None, ok=lambda *a, **k: True)
    mod("rclpy.time", Time=Any)
    mod("rclpy.duration", Duration=Any)
    mod("rclpy.node", Node=type("Node", (), {"__init__": lambda s, *a, **k: None}))
    mod("rclpy.action", ActionClient=Any)
    mod("rclpy.callback_groups", ReentrantCallbackGroup=Any)
    mod("rclpy.executors", MultiThreadedExecutor=Any)
    for n in ("moveit_msgs", "moveit_msgs.msg", "moveit_msgs.srv", "moveit_msgs.action",
              "control_msgs", "control_msgs.action", "geometry_msgs", "geometry_msgs.msg",
              "sensor_msgs", "sensor_msgs.msg", "shape_msgs", "shape_msgs.msg",
              "trajectory_msgs", "trajectory_msgs.msg", "nav_msgs", "nav_msgs.msg",
              "tf2_ros", "tf2_geometry_msgs", "sensor_msgs_py"):
        m = mod(n)
        m.__getattr__ = lambda name: Any            # type: ignore[attr-defined]
        for sym in ("CollisionObject", "MoveItErrorCodes", "Constraints", "JointConstraint",
                    "RobotState", "PositionConstraint", "OrientationConstraint",
                    "BoundingVolume", "PlanningScene", "DisplayTrajectory", "GetPositionIK",
                    "GetPositionFK", "GetPlanningScene", "GetCartesianPath",
                    "ApplyPlanningScene", "MoveGroup", "ExecuteTrajectory", "GripperCommand",
                    "Pose", "PointStamped", "PoseStamped", "JointState", "PointCloud2",
                    "SolidPrimitive", "JointTrajectory", "Path", "TransformListener",
                    "Buffer", "do_transform_point"):
            setattr(m, sym, Any)

    # the node imports `from sensor_msgs_py import point_cloud2`
    pc2 = mod("sensor_msgs_py.point_cloud2")
    sys.modules["sensor_msgs_py"].point_cloud2 = pc2
    return pc2


PC2 = _stub_ros()


def load_node_class():
    import importlib.util
    src = os.path.join(ROOT, "reactive_replanning_ur12e", "reactive_replanning.py")
    spec = importlib.util.spec_from_file_location("reactive_replanning", src)
    m = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):      # module prints on import
        spec.loader.exec_module(m)
    return m.ReactiveReplannerUR12e


def mat_to_quat(R):
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return types.SimpleNamespace(x=x, y=y, z=z, w=w)


class _Tf:
    """Answers lookup_transform for the camera and every arm link."""
    def __init__(self, links):
        self.links = links
        self.cam_q = mat_to_quat(scene.R_BASE_FROM_CAM)

    def lookup_transform(self, target, source, *a, **k):
        assert target == 'base_link'
        if source in self.links:
            p, q = self.links[source], types.SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)
        else:                                            # camera optical frame
            p, q = scene.CAM_POS, self.cam_q
        return types.SimpleNamespace(transform=types.SimpleNamespace(
            translation=types.SimpleNamespace(x=p[0], y=p[1], z=p[2]), rotation=q))


class Rig:
    """A bare node with the pipeline's state wired up, plus detection capture."""

    def __init__(self, cls, links=None):
        from collections import deque
        n = object.__new__(cls)
        n.tf_buffer = _Tf(links or scene.LINKS)
        n._camera_ok = False
        n._last_cloud_t = 0.0
        n._baseline_samples = []
        n._baseline_count = None
        n._executing = False
        n._obstacle_present = False
        n._obstacle_streak = 0
        n._last_obstacle_seen = 0.0
        n.obstacle_detected = False
        n._arm_pos_history = deque(maxlen=3)
        n._inject_obstacle_at_xyz = self._inject
        n._remove_obstacle = self._remove
        self.node = n
        self.detections = []

    def _inject(self, x, y, z, diff):
        self.detections.append((np.array([x, y, z]), diff))
        self.node._obstacle_present = True
        return True

    def _remove(self):
        self.node._obstacle_present = False

    def feed(self, xyz_cam, rgb):
        """One camera frame through the real callback. Returns elapsed seconds."""
        import time
        rec = np.zeros(len(xyz_cam), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('rgb', 'f4')])
        rec['x'], rec['y'], rec['z'] = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
        packed = ((rgb[:, 0].astype(np.uint32) << 16) |
                  (rgb[:, 1].astype(np.uint32) << 8) | rgb[:, 2].astype(np.uint32))
        rec['rgb'] = packed.view(np.float32) if packed.dtype == np.uint32 else packed
        PC2.read_points_numpy = lambda msg, field_names=None, skip_nans=True: rec

        msg = types.SimpleNamespace(header=types.SimpleNamespace(frame_id='camera_depth_optical_frame'))
        self.node._last_cloud_t = 0.0                    # bypass the 20 Hz throttle
        t0 = time.perf_counter()
        self.node._cloud_cb(msg)
        return time.perf_counter() - t0
