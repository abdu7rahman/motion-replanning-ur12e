"""Pick and place as a phase machine, and what counts as success.

Success is the object on the place table. Not "the arm survived", which is what
an avoidance-only trial measures and which a robot that stops dead also
achieves. A run that dodges the obstacle perfectly and drops the cube on the
floor has failed at the job.

The phases below are TCP targets, not wrist targets. Solving IK against the
wrist origin instead puts the Hand-E 0.157 m lower than asked, which is far
enough to drive the fingers through a 0.30 m table -- that was a real bug here,
found by listing MuJoCo's penetrating contacts along the nominal path rather
than by looking at the render, where it is invisible.

Only the free-space part of the motion may be deformed by the replanner. Lifting
away from the bench and carrying across are where an obstacle matters and where
there is room to move; the descent onto the cube, the grasp and the release are
precision moves against a known table, and letting an obstacle-avoidance push
perturb them would trade the task for the dodge.
"""
from __future__ import annotations

__author__ = "".join(
    chr(c - 7) for c in (104, 105, 107, 124, 115, 39, 121, 104, 111, 116, 104, 117)
)

from dataclasses import dataclass

import numpy as np

from predictive_replanning.assets import HANDE
from predictive_replanning.cell import CUBE_HALF, CUBE_XY, PICK_TABLE, PLACE_TABLE, PLACE_XY
from predictive_replanning.ur12e import TOP_DOWN, fk_tcp_pos, ik_pose

# Hand-E finger travel is 0 .. 0.025 m per finger, and the jaw gap measured off
# the vendor's own collision meshes is exactly twice it: travel 0 is CLOSED and
# 0.025 is a 50 mm opening. An earlier version had these the wrong way round,
# which is half of why nothing could be picked up -- the other half being that
# the fingers carried no collision geometry at all.
#
# So the commands are derived from the object, not typed in. The jaws touch a
# box of width w at travel w/2; open clears it by GRIP_CLEARANCE and the grasp
# command sits GRIP_SQUEEZE inside the faces, which with a force-limited
# actuator is what produces grip force instead of penetration.
GRIP_SPAN_PER_TRAVEL = 2.0
GRIP_CLEARANCE = 0.008          # m of gap either side when open
GRIP_SQUEEZE = HANDE["grip_squeeze_m"]   # m the command goes inside each face


def grip_open(width: float) -> float:
    return min(0.025, width / GRIP_SPAN_PER_TRAVEL + GRIP_CLEARANCE)


def grip_grasp(width: float) -> float:
    return max(0.0, width / GRIP_SPAN_PER_TRAVEL - GRIP_SQUEEZE)


@dataclass
class Phase:
    name: str
    tcp: np.ndarray
    grip: float               # commanded finger travel, metres
    seconds: float
    deformable: bool          # may the replanner move this stretch?
    holding: bool             # is the cube expected to be in the jaws?


def pick_and_place(cube: int = 0) -> list[Phase]:
    """The eight-phase sequence, in base_link with the pedestal removed."""
    base_z = 0.18
    pick_top = PICK_TABLE["centre"][2] + PICK_TABLE["half"][2]
    place_top = PLACE_TABLE["centre"][2] + PLACE_TABLE["half"][2]
    cx, cy = CUBE_XY[cube]
    px, py = PLACE_XY

    def at(x, y, z):
        return np.array([x, y, z - base_z])

    grasp_z = pick_top + CUBE_HALF          # jaws centred on the cube
    width = 2.0 * CUBE_HALF
    op, cl = grip_open(width), grip_grasp(width)
    return [
        Phase("approach", at(cx, cy, pick_top + 0.16), op, 1.6, True, False),
        Phase("descend", at(cx, cy, grasp_z), op, 1.4, False, False),
        # The jaws close here and the arm settles before anything is lifted.
        Phase("close", at(cx, cy, grasp_z), cl, 1.0, False, False),
        Phase("lift", at(cx, cy, pick_top + 0.24), cl, 1.4, True, True),
        Phase("transfer", at(px, py, place_top + 0.24), cl, 3.0, True, True),
        Phase("place", at(px, py, place_top + CUBE_HALF + 0.006), cl, 1.6, False, True),
        Phase("release", at(px, py, place_top + CUBE_HALF + 0.006), op, 0.8, False, False),
        Phase("retreat", at(px, py, place_top + 0.20), op, 1.2, False, False),
    ]


def solve(phases: list[Phase], q_home, *, per_phase: int = 14):
    """Chain the phases into one joint trajectory.

    Returns (traj, times, grip, deformable, holding) sampled at the same rate,
    so the controller reads one index and gets every command for that instant.
    """
    q = np.array(q_home, dtype=float)
    tcp = fk_tcp_pos(q)
    traj, times, grip, deform, holding = [], [], [], [], []
    t = 0.0
    for ph in phases:
        u = np.linspace(0.0, 1.0, per_phase)
        s = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5      # quintic, zero end rates
        for k, sk in enumerate(s):
            # Interpolate the TOOL along a straight line and solve for the arm
            # at every waypoint, rather than interpolating joint angles between
            # two solved poses. Joint-space interpolation only pins the tool at
            # the ends: in between it swung up to 9.8 degrees off vertical, so
            # the "top-down" grasp was top-down twice per phase and tilted the
            # rest of the time.
            target = tcp + sk * (np.asarray(ph.tcp, float) - tcp)
            q, ok, pe, re = ik_pose(target, TOP_DOWN, q)
            if not ok:
                raise RuntimeError(
                    f"IK failed in phase {ph.name} at s={sk:.2f}: "
                    f"{pe:.4f} m, {re:.4f} rad")
            traj.append(q.copy())
            times.append(t + u[k] * ph.seconds)
            grip.append(ph.grip)
            deform.append(ph.deformable)
            holding.append(ph.holding)
        t += ph.seconds
        tcp = np.asarray(ph.tcp, float)
    return (np.asarray(traj), np.asarray(times), np.asarray(grip),
            np.asarray(deform, dtype=bool), np.asarray(holding, dtype=bool))


def placed(cube_pos, *, xy_tol: float = 0.06) -> tuple[bool, float, float]:
    """Is the cube on the place table where it was asked to go?

    Returns (ok, xy_error, height_error). Both conditions matter: a cube nudged
    off the edge lands at the right xy and the wrong height, and one left on the
    pick table never moved at all.
    """
    place_top = PLACE_TABLE["centre"][2] + PLACE_TABLE["half"][2]
    target_z = place_top + CUBE_HALF
    xy_err = float(np.hypot(cube_pos[0] - PLACE_XY[0], cube_pos[1] - PLACE_XY[1]))
    dz = float(abs(cube_pos[2] - target_z))
    return bool(xy_err < xy_tol and dz < 0.03), xy_err, dz
