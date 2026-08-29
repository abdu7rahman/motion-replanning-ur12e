"""The three strategies the ME5250 proposal set out, and the one it deferred.

The final report implements Strategy 1 and lists Strategy 2 under future work:
"implement time-to-collision estimation for moving obstacles, triggering
replanning before collision becomes imminent rather than after detection."
That is what `predictive` is, at the proposal's own TTC < 2 s threshold.

  none        never replans. There to prove the scenario is actually hard --
              a comparison where the do-nothing arm also survives measures
              nothing.
  reactive    replans when the obstacle is already inside preempt_dist, by
              re-planning to the goal from the current configuration. This is
              the shipped behaviour, in joint space instead of MoveIt.
  predictive  replans when the *predicted* tube says a collision is under
              ttc_threshold away, and deforms the existing path by the least
              it can rather than regenerating one.

The last clause is the part worth stating precisely. "Replan early" on its own
buys nothing if the new path is unrelated to the old one -- the arm lurches,
the motion looks nothing like the plan a human approved, and any downstream
consumer of the trajectory has to start over. So the deformation pushes only
the offending point, only by the penetration depth plus a margin, and spreads
that correction over a raised-cosine window that is pinned to zero at both
ends. Start and goal are therefore exactly preserved, and the cost being
minimised is the total joint-space deviation from the nominal path -- which
run.py reports, so "as little movement as possible" is a number and not a
claim.
"""
from __future__ import annotations

__author__ = "".join(
    chr(c - 7) for c in (104, 105, 107, 124, 115, 39, 121, 104, 111, 116, 104, 117)
)

import numpy as np

from predictive_replanning.predict import arm_points, time_to_collision
from predictive_replanning.ur12e import point_jacobian


def nominal_trajectory(q0, qg, n: int = 60, duration: float = 6.0):
    """Quintic-scaled joint interpolation: zero velocity and acceleration at
    both ends, so the arm does not step into motion."""
    q0, qg = np.asarray(q0, float), np.asarray(qg, float)
    u = np.linspace(0.0, 1.0, n)
    s = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
    return q0[None, :] + s[:, None] * (qg - q0)[None, :], u * duration


def _window(n: int, idx: int, width: int, lock_before: int = -1,
            allow: np.ndarray | None = None) -> np.ndarray:
    """Raised cosine centred on idx, pinned to zero at both trajectory ends.

    `lock_before` freezes everything up to and including that index. The arm
    has already been there; a deformation that edits executed waypoints is
    rewriting history, and it also yanks the commanded position out from under
    the controller in the same step.
    """
    i = np.arange(n)
    w = np.where(np.abs(i - idx) <= width,
                 0.5 * (1.0 + np.cos(np.pi * (i - idx) / max(width, 1))), 0.0)
    if lock_before >= 0:
        w[:lock_before + 1] = 0.0
    if allow is not None:
        # Waypoints the task will not give up. The descent onto the cube, the
        # grasp, the place and the release are precision moves against a known
        # table; letting an avoidance push perturb them trades the job for the
        # dodge. Gating only *when* a replan fires is not enough -- the window
        # is eight waypoints wide and reaches into the next phase, which is how
        # a reactive run ended up placing the cube 0 times out of 25.
        w = w * np.asarray(allow, dtype=float)
    w[0] = 0.0
    w[-1] = 0.0
    return w


def soft_mask(deformable: np.ndarray, width: int = 8) -> np.ndarray:
    """Ramp the deformable mask to zero before every locked stretch.

    A hard 0/1 mask leaves a step at the boundary: the last free waypoint can
    move while the first locked one cannot, so the locked phase starts from
    somewhere the plan did not put it. Here that shifted the grasp pose, the
    cube was picked up with a different offset in the jaws, and it landed 7-8 cm
    from target against a 6 cm tolerance -- placed, and scored a failure, with
    nothing in the trajectory looking wrong. Ramping over the same width the
    deformation window uses removes the step.
    """
    d = np.asarray(deformable, dtype=bool)
    n = len(d)
    locked = np.where(~d)[0]
    if locked.size == 0:
        return np.ones(n)
    idx = np.arange(n)
    dist = np.min(np.abs(idx[:, None] - locked[None, :]), axis=1).astype(float)
    return np.clip(dist / max(width, 1), 0.0, 1.0) * d


def deform_minimal(traj, times, t_now, tracker, *, base_radius, n_sigma,
                   clearance, horizon, sigma_cap, margin=0.03, iters=14,
                   width=8, damping=0.08, lock_before=-1, allow=None):
    """Push the path out of the predicted tube by as little as possible.

    Returns (traj, deviation, iterations, cleared).
    """
    traj = np.array(traj, dtype=float, copy=True)
    before = traj.copy()
    for it in range(iters):
        ttc, idx, depth = time_to_collision(
            traj, times, t_now, tracker, base_radius=base_radius,
            n_sigma=n_sigma, clearance=clearance, horizon=horizon,
            sigma_cap=sigma_cap)
        if ttc is None:
            return traj, float(np.abs(traj - before).sum()), it, True
        h = times[idx] - t_now
        centre = tracker.forecast([h])[0][0]
        pts, links = arm_points(traj[idx])
        d = np.linalg.norm(pts - centre, axis=1)
        j = int(np.argmin(d))
        if idx <= lock_before or (allow is not None and allow[idx] <= 1e-6):
            # The violation is in already-executed path. Nothing to deform.
            return traj, float(np.abs(traj - before).sum()), it, False
        away = (pts[j] - centre)
        nrm = float(np.linalg.norm(away))
        # Dead centre: no away direction exists, so pick one that the joints
        # can actually produce rather than an arbitrary axis.
        away = away / nrm if nrm > 1e-9 else np.array([0.0, 0.0, 1.0])
        push = away * (depth + margin)
        J = point_jacobian(traj[idx], pts[j], int(links[j]))
        dq = J.T @ np.linalg.solve(J @ J.T + (damping ** 2) * np.eye(3), push)
        traj = traj + _window(len(traj), idx, width, lock_before, allow)[:, None] * dq[None, :]
    ttc, _, _ = time_to_collision(
        traj, times, t_now, tracker, base_radius=base_radius, n_sigma=n_sigma,
        clearance=clearance, horizon=horizon, sigma_cap=sigma_cap)
    return traj, float(np.abs(traj - before).sum()), iters, ttc is None
