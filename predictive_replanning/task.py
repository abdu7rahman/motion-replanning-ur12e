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

from predictive_replanning.cell import CUBE_HALF, CUBE_XY, PICK_TABLE, PLACE_TABLE, PLACE_XY
from predictive_replanning.ur12e import ik_tcp

GRIP_OPEN, GRIP_CLOSED = 0.0, 0.022      # Hand-E travel is 0 .. 0.025 m


@dataclass
class Phase:
    name: str
    tcp: np.ndarray
    grip: float
    seconds: float
    deformable: bool          # may the replanner move this stretch?
    weld: bool                # is the cube attached during it?


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
    return [
        Phase("approach", at(cx, cy, pick_top + 0.16), GRIP_OPEN, 1.6, True, False),
        Phase("descend", at(cx, cy, grasp_z), GRIP_OPEN, 1.2, False, False),
        # weld=False here on purpose. The jaws close during this phase and the
        # arm settles; engaging the attachment on its first waypoint captured
        # the cube while the controller was still 0.075 rad from the commanded
        # pose, and that offset rode all the way through to a placement 4 cm
        # off target -- most of the 6 cm budget spent before the carry even
        # started. The attachment engages at the lift instead.
        Phase("grasp", at(cx, cy, grasp_z), GRIP_CLOSED, 1.0, False, False),
        Phase("lift", at(cx, cy, pick_top + 0.24), GRIP_CLOSED, 1.2, True, True),
        Phase("transfer", at(px, py, place_top + 0.24), GRIP_CLOSED, 3.0, True, True),
        Phase("place", at(px, py, place_top + CUBE_HALF + 0.004), GRIP_CLOSED, 1.4, False, True),
        Phase("release", at(px, py, place_top + CUBE_HALF + 0.004), GRIP_OPEN, 0.6, False, False),
        Phase("retreat", at(px, py, place_top + 0.20), GRIP_OPEN, 1.2, False, False),
    ]


def solve(phases: list[Phase], q_home, *, per_phase: int = 14):
    """Chain the phases into one joint trajectory.

    Returns (traj, times, grip, deformable, weld) sampled at the same rate, so
    the controller reads one index and gets every command for that instant.
    """
    q = np.array(q_home, dtype=float)
    traj, times, grip, deform, weld = [], [], [], [], []
    t = 0.0
    for ph in phases:
        q_goal, ok, err = ik_tcp(ph.tcp, q)
        if not ok:
            raise RuntimeError(f"IK failed for phase {ph.name}: residual {err:.4f} m")
        u = np.linspace(0.0, 1.0, per_phase)
        s = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5      # quintic, zero end rates
        for k, sk in enumerate(s):
            traj.append(q + sk * (q_goal - q))
            times.append(t + u[k] * ph.seconds)
            grip.append(ph.grip)
            deform.append(ph.deformable)
            weld.append(ph.weld)
        t += ph.seconds
        q = q_goal
    return (np.asarray(traj), np.asarray(times), np.asarray(grip),
            np.asarray(deform, dtype=bool), np.asarray(weld, dtype=bool))


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
