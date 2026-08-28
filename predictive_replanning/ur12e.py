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
