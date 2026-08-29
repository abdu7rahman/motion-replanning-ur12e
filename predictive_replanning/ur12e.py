"""UR12e kinematics, from the joint origins the description package ships.

The six numbers below are `config/ur12e/default_kinematics.yaml` in
Universal_Robots_ROS2_Description, copied rather than re-derived. That file is
byte-identical to `config/ur10e/default_kinematics.yaml` -- the UR12e is a
payload variant of the UR10e and shares its geometry exactly. It is worth
saying out loud because UR's public DH table does not list the UR12e at all
(UR3e, UR5e, UR10e, UR16e, UR20, UR30 only), so the obvious place to look
comes up empty and the obvious next move is to guess. `config/ur16e` differs
(a2 = -0.4784), which is what makes the identical UR10e/UR12e files a
statement about the robots rather than about the repository.

Link collision geometry is NOT from this source. UR ships meshes; capsules
sized to the joint offsets stand in for them here, which is fine for a
replanning study -- the arm still sweeps the right volume to within a
centimetre -- and wrong for anything that needs the real hull.
"""
from __future__ import annotations

__author__ = "".join(
    chr(c - 7) for c in (104, 105, 107, 124, 115, 39, 121, 104, 111, 116, 104, 117)
)

import numpy as np

#: (x, y, z, roll, pitch, yaw) per joint, parent -> child, before the joint's
#: own rotation about its local Z. Verbatim from default_kinematics.yaml.
JOINT_ORIGINS: tuple[tuple[float, ...], ...] = (
    (0.0,      0.0,      0.1807,   0.0,            0.0,            0.0),
    (0.0,      0.0,      0.0,      1.570796327,    0.0,            0.0),
    (-0.6127,  0.0,      0.0,      0.0,            0.0,            0.0),
    (-0.57155, 0.0,      0.17415,  0.0,            0.0,            0.0),
    (0.0,     -0.11985,  0.0,      1.570796327,    0.0,            0.0),
    (0.0,      0.11655,  0.0,      1.570796326589793,
                                   3.141592653589793,
                                   3.141592653589793),
)

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow",
               "wrist_1", "wrist_2", "wrist_3")

#: From this cell's own config/moveit/joint_limits.yaml, not from upstream.
VEL_LIMITS = np.array([2.0944, 2.0944, 3.1416, 3.1416, 3.1416, 3.1416])
ACC_LIMITS = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])

#: The z origins are ~2.4e-11, i.e. zero written by a calibration dump. Kept in
#: JOINT_ORIGINS as 0.0; recording the real value here so the rounding is a
#: decision on the record rather than a silent one.
CALIB_Z_EPSILON = 2.458164590756244e-11


def rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis XYZ: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def _origin(i: int) -> np.ndarray:
    x, y, z, r, p, yw = JOINT_ORIGINS[i]
    T = np.eye(4)
    T[:3, :3] = rpy(r, p, yw)
    T[:3, 3] = (x, y, z)
    return T


_ORIGINS = tuple(_origin(i) for i in range(6))


def _rotz(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    T = np.eye(4)
    T[0, 0], T[0, 1], T[1, 0], T[1, 1] = c, -s, s, c
    return T


def link_frames(q) -> list[np.ndarray]:
    """Every joint frame from base to wrist_3, as 4x4 in base_link."""
    q = np.asarray(q, dtype=float)
    T = np.eye(4)
    out = []
    for i in range(6):
        T = T @ _ORIGINS[i] @ _rotz(q[i])
        out.append(T.copy())
    return out


def fk(q) -> np.ndarray:
    """Pose of wrist_3_link in base_link."""
    return link_frames(q)[-1]


def fk_pos(q) -> np.ndarray:
    return fk(q)[:3, 3]


def jacobian(q) -> np.ndarray:
    """Geometric Jacobian (6x6) at wrist_3, in base_link.

    Every joint is revolute about its own local z, so the column is the usual
    (z_i x (p_e - p_i), z_i) pair. Built from the same frames FK returns, so a
    disagreement between the two is impossible by construction.
    """
    frames = link_frames(q)
    p_e = frames[-1][:3, 3]
    J = np.zeros((6, 6))
    for i, T in enumerate(frames):
        z = T[:3, 2]
        J[:3, i] = np.cross(z, p_e - T[:3, 3])
        J[3:, i] = z
    return J


def ik(target_pos, q_seed, *, iters: int = 200, tol: float = 1e-4,
       damping: float = 0.05, pos_only: bool = True):
    """Damped least squares to a Cartesian position, seeded at `q_seed`.

    Position-only by default: this cell's task is pick-and-place with a
    top-down gripper, and leaving orientation free is what gives the replanner
    somewhere to go. Returns (q, converged, residual).
    """
    q = np.array(q_seed, dtype=float)
    target = np.asarray(target_pos, dtype=float)
    err = np.inf
    for _ in range(iters):
        p = fk_pos(q)
        e = target - p
        err = float(np.linalg.norm(e))
        if err < tol:
            return q, True, err
        J = jacobian(q)[:3] if pos_only else jacobian(q)
        JT = J.T
        # (J J^T + k^2 I)^-1 keeps the step finite through the shoulder and
        # wrist singularities, where the plain pseudo-inverse blows up.
        dq = JT @ np.linalg.solve(J @ JT + (damping ** 2) * np.eye(J.shape[0]), e)
        step = float(np.linalg.norm(dq))
        if step > 0.25:                      # cap so a far target cannot fling
            dq *= 0.25 / step
        q = q + dq
    return q, err < tol, err


def point_jacobian(q, point, link: int) -> np.ndarray:
    """Translational Jacobian (3x6) of a point riding on `link`.

    Joints past that link do not move it, so their columns are zero -- which is
    what stops a correction meant for the forearm from being applied at the
    wrist, where it would rotate the tool and move nothing.
    """
    frames = link_frames(q)
    p = np.asarray(point, dtype=float)
    J = np.zeros((3, 6))
    for i in range(min(link, 6)):
        T = frames[i]
        J[:, i] = np.cross(T[:3, 2], p - T[:3, 3])
    return J


# ── the tool chain, wrist_3 -> TCP ────────────────────────────────────
# wrist_3 -> flange and flange -> tool0 are both fixed joints in
# ur_macro.xacro; the two Hand-E offsets are xacro argument defaults in
# robotiq_hande_description. Composed once at import.
_FLANGE_RPY = (0.0, -np.pi / 2.0, -np.pi / 2.0)
_TOOL0_RPY = (np.pi / 2.0, 0.0, np.pi / 2.0)
#: coupler_height 0.011 + hande_height 0.099 + the end offset 0.0465.
TCP_OFFSET_Z = 0.011 + 0.099 + 0.0465


def _tool_transform() -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rpy(*_FLANGE_RPY) @ rpy(*_TOOL0_RPY)
    T = T @ np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, TCP_OFFSET_Z], [0, 0, 0, 1.0]])
    return T


_TOOL = _tool_transform()


def fk_tcp(q) -> np.ndarray:
    """Pose of the Hand-E's tool centre point, in base_link.

    Solving IK against wrist_3 instead of this puts the gripper 0.157 m below
    where it was asked to go, which is exactly far enough to drive the fingers
    through a 0.30 m table.
    """
    return fk(q) @ _TOOL


def fk_tcp_pos(q) -> np.ndarray:
    return fk_tcp(q)[:3, 3]


def tcp_jacobian(q) -> np.ndarray:
    """Translational Jacobian of the TCP: the wrist Jacobian carried out to the
    tool point, so the moment arm of the gripper is accounted for."""
    frames = link_frames(q)
    p_e = fk_tcp_pos(q)
    J = np.zeros((3, 6))
    for i, T in enumerate(frames):
        J[:, i] = np.cross(T[:3, 2], p_e - T[:3, 3])
    return J


def ik_tcp(target_pos, q_seed, *, iters: int = 300, tol: float = 1e-4,
           damping: float = 0.05):
    """Damped least squares onto a TCP position, seeded at `q_seed`."""
    q = np.array(q_seed, dtype=float)
    target = np.asarray(target_pos, dtype=float)
    err = np.inf
    for _ in range(iters):
        e = target - fk_tcp_pos(q)
        err = float(np.linalg.norm(e))
        if err < tol:
            return q, True, err
        J = tcp_jacobian(q)
        dq = J.T @ np.linalg.solve(J @ J.T + (damping ** 2) * np.eye(3), e)
        step = float(np.linalg.norm(dq))
        if step > 0.25:
            dq *= 0.25 / step
        q = q + dq
    return q, err < tol, err


def tcp_jacobian_full(q) -> np.ndarray:
    """6x6 Jacobian at the TCP: linear rows then angular rows."""
    frames = link_frames(q)
    p_e = fk_tcp_pos(q)
    J = np.zeros((6, 6))
    for i, T in enumerate(frames):
        z = T[:3, 2]
        J[:3, i] = np.cross(z, p_e - T[:3, 3])
        J[3:, i] = z
    return J


def _log_so3(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis-angle vector, the orientation error term."""
    c = (np.trace(R) - 1.0) / 2.0
    c = float(np.clip(c, -1.0, 1.0))
    ang = np.arccos(c)
    if ang < 1e-9:
        return np.zeros(3)
    if abs(np.pi - ang) < 1e-6:
        # near pi the skew part vanishes; take the axis from R + I instead
        w = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        k = int(np.argmax(w))
        axis = (R[:, k] + np.eye(3)[:, k]) / (2.0 * w[k])
        return ang * axis / np.linalg.norm(axis)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return ang * v / (2.0 * np.sin(ang))


def ik_pose(target_pos, target_R, q_seed, *, iters: int = 400, pos_tol: float = 5e-4,
            rot_tol: float = 0.01, damping: float = 0.06, rot_weight: float = 0.6):
    """Full 6-DOF damped least squares onto a TCP pose.

    Position-only IK was a real modelling error rather than a simplification.
    Minimum-norm steps move whichever joints shift the tool most cheaply, so the
    shoulder did nearly all the work, the elbow swung 0.27 rad against the
    shoulder's 0.93, and wrist_3 never moved at all -- it cannot change a
    position, so a position-only task gives it nothing to do. The visible result
    is an arm that slides around with a locked elbow instead of articulating.

    Worse, nothing held the tool upright: the gripper arrived at the cube 36.6
    degrees off vertical and drifted between 21 and 37 degrees across the task.
    The jaws straddled the cube by luck. Commanding the orientation makes the
    pick top-down, gives the wrist something to do, and makes the arm move the
    way the hardware would.

    `rot_weight` trades the two errors; orientation is in radians and position
    in metres, so they are not comparable without it.
    """
    q = np.array(q_seed, dtype=float)
    target_pos = np.asarray(target_pos, dtype=float)
    target_R = np.asarray(target_R, dtype=float)
    pe = re = np.inf
    for _ in range(iters):
        T = fk_tcp(q)
        e_p = target_pos - T[:3, 3]
        e_r = _log_so3(target_R @ T[:3, :3].T)
        pe, re = float(np.linalg.norm(e_p)), float(np.linalg.norm(e_r))
        if pe < pos_tol and re < rot_tol:
            return q, True, pe, re
        e = np.concatenate([e_p, rot_weight * e_r])
        J = tcp_jacobian_full(q)
        J = np.vstack([J[:3], rot_weight * J[3:]])
        dq = J.T @ np.linalg.solve(J @ J.T + (damping ** 2) * np.eye(6), e)
        step = float(np.linalg.norm(dq))
        if step > 0.2:
            dq *= 0.2 / step
        q = q + dq
    return q, (pe < pos_tol and re < rot_tol), pe, re


#: Tool pose for a top-down grasp: tool z straight down, jaw axis along world X.
#: The Hand-E's fingers slide on the tool x axis, so this is what decides which
#: way the jaws close across the object.
TOP_DOWN = np.array([[1.0, 0.0, 0.0],
                     [0.0, -1.0, 0.0],
                     [0.0, 0.0, -1.0]])


def point_jacobian_full(q, point, link: int) -> np.ndarray:
    """6xN for a point on `link`: its translation rows, and the TCP's rotation rows.

    The rotation rows are what let a caller push the point while commanding the
    tool to hold its orientation. A purely translational push is free to tilt
    the wrist, and tilting a gripper mid-carry is how a friction grasp loses
    the object it is holding.
    """
    frames = link_frames(q)
    p = np.asarray(point, dtype=float)
    J = np.zeros((6, 6))
    for i in range(6):
        z = frames[i][:3, 2]
        if i < min(link, 6):
            J[:3, i] = np.cross(z, p - frames[i][:3, 3])
        J[3:, i] = z
    return J
