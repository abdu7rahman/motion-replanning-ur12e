"""Synthetic RealSense scenes for exercising the detection pipeline.

Builds point clouds in the camera optical frame containing a table, the arm as
a chain of capsules, and optionally a foreign object at a known base_link
position -- so every detection can be scored against ground truth.
"""
import numpy as np

# UR12e-ish arm pose in base_link. Fixed so runs are comparable.
LINKS = {
    'base_link':      np.array([0.00,  0.00, 0.00]),
    'shoulder_link':  np.array([0.00,  0.00, 0.18]),
    'upper_arm_link': np.array([-0.15, 0.05, 0.40]),
    'forearm_link':   np.array([-0.45, 0.08, 0.55]),
    'wrist_1_link':   np.array([-0.68, 0.08, 0.45]),
    'wrist_2_link':   np.array([-0.72, 0.08, 0.36]),
    'wrist_3_link':   np.array([-0.74, 0.08, 0.30]),
    'tool0':          np.array([-0.76, 0.08, 0.25]),
}
CHAIN = [('base_link', 'shoulder_link'), ('shoulder_link', 'upper_arm_link'),
         ('upper_arm_link', 'forearm_link'), ('forearm_link', 'wrist_1_link'),
         ('wrist_1_link', 'wrist_2_link'), ('wrist_2_link', 'wrist_3_link'),
         ('wrist_3_link', 'tool0')]

# Camera looks at the workspace from the far side; base_link -> camera.
CAM_POS = np.array([-0.70, -1.15, 0.95])
GRAY   = (185, 188, 190)   # UR housing
BLACK  = (18, 18, 20)      # accents
TEAL   = (90, 165, 175)    # trim
SKIN   = (200, 145, 115)   # a hand
TABLE  = (150, 152, 155)


def _cam_basis():
    fwd = np.array([0.0, 0.0, 0.0]) - CAM_POS
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0])); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    # camera optical frame: x right, y down, z forward
    return np.stack([right, down, fwd])          # rows = optical axes in base


R_BASE_FROM_CAM = _cam_basis().T                  # base_link <- camera


def to_camera(pts_base):
    return (pts_base - CAM_POS) @ R_BASE_FROM_CAM


def capsule(a, b, radius, n, rng):
    t = rng.random(n)[:, None]
    axis = a + t * (b - a)
    d = rng.normal(size=(n, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    return axis + d * radius * rng.random(n)[:, None] ** (1 / 3)


def sphere(c, radius, n, rng):
    d = rng.normal(size=(n, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    return c + d * radius * rng.random(n)[:, None] ** (1 / 3)


def build(rng, obstacle=None, obstacle_pts=900, obstacle_rgb=SKIN,
          arm_density=170, table_pts=2600, noise=0.004):
    """Returns (xyz_camera, rgb, ground_truth_centre_or_None)."""
    pts, cols = [], []

    # table slab inside the workspace box
    tx = rng.uniform(-1.10, -0.30, table_pts)
    ty = rng.uniform(-0.45, 0.55, table_pts)
    tz = np.full(table_pts, 0.12) + rng.normal(0, 0.002, table_pts)
    pts.append(np.stack([tx, ty, tz], axis=1)); cols.append(np.tile(TABLE, (table_pts, 1)))

    # arm: capsules along the chain, mixed housing colours
    for a_name, b_name in CHAIN:
        a, b = LINKS[a_name], LINKS[b_name]
        seg = max(1, int(arm_density * (np.linalg.norm(b - a) + 0.05)))
        p = capsule(a, b, 0.055, seg, rng)
        c = rng.random(seg)
        col = np.where(c[:, None] < 0.72, GRAY, np.where(c[:, None] < 0.9, BLACK, TEAL))
        pts.append(p); cols.append(col)

    gt = None
    if obstacle is not None:
        centre, radius = obstacle
        p = sphere(np.asarray(centre), radius, obstacle_pts, rng)
        pts.append(p); cols.append(np.tile(obstacle_rgb, (obstacle_pts, 1)))
        gt = np.asarray(centre, dtype=float)

    xyz = np.vstack(pts).astype(np.float64)
    rgb = np.vstack(cols).astype(np.int16)
    xyz = xyz + rng.normal(0, noise, xyz.shape)
    return to_camera(xyz), rgb, gt
