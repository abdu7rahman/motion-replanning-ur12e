"""Tracking the obstacle, and turning the track into a time-to-collision.

The tracker is a constant-velocity Kalman filter over [x, y, z, vx, vy, vz]
watching noisy positions. CV is deliberately the wrong model -- the truth is
OU velocity, which decays toward the mean -- and that mismatch is the point:
the robot does not get told the process that generates the motion, it has to
estimate it. What the filter does carry correctly is the *growth* of its own
uncertainty, and that is what widens the tube.

Prediction is therefore not a point. At horizon h the obstacle is a sphere of

    r_eff(h) = r_obstacle + n_sigma * sigma_pos(h)

so avoiding it means avoiding a cone that opens with time. Checking only the
predicted mean would make the robot exactly as brittle as the reactive baseline
it is supposed to beat, just earlier.

Collision is checked against sampled points along the arm, not the tool centre
alone: the forearm sweeps a much larger volume than the gripper, and an elbow
that clears nothing while the TCP clears everything is the failure this whole
exercise is about.
"""
from __future__ import annotations

__author__ = "".join(
    chr(c - 7) for c in (104, 105, 107, 124, 115, 39, 121, 104, 111, 116, 104, 117)
)

import numpy as np

from predictive_replanning.ur12e import fk_tcp_pos, link_frames


def arm_points(q, per_link: int = 3):
    """Points along the arm *including the gripper*, with their link index.

    The index is what lets the replanner build a Jacobian for the point that is
    actually in the way: pushing the tool when the forearm is inside the
    obstacle moves the wrong part of the arm.

    The chain runs to the tool centre point, not to wrist_3. Stopping at the
    wrist left the last 0.156 m of the robot -- the whole Hand-E, and whatever
    it is holding -- invisible to the planner. That is the part furthest from
    the base and nearest the obstacle during a carry, so the replanner was
    blind exactly where it mattered and barely triggered at all: 1.2 replans
    per run, with the path 1.002x nominal, against an obstacle sitting in the
    corridor. The gripper rides on link 6, so its Jacobian columns are the full
    six and a push applied there moves it.
    """
    frames = link_frames(q)
    pts = [np.zeros(3)] + [T[:3, 3] for T in frames] + [fk_tcp_pos(q)]
    out, links = list(pts), list(range(len(pts) - 1)) + [6]
    for j, (a, b) in enumerate(zip(pts, pts[1:])):
        for s in np.linspace(0.0, 1.0, per_link + 2)[1:-1]:
            out.append(a + s * (b - a))
            links.append(min(j + 1, 6))
    return np.asarray(out), np.asarray(links)


class ObstacleTracker:
    """Constant-velocity Kalman filter with an explicit forward covariance."""

    def __init__(self, pos, *, meas_std: float = 0.02, accel_std: float = 1.2):
        self.x = np.concatenate([np.asarray(pos, float), np.zeros(3)])
        self.P = np.diag([meas_std ** 2] * 3 + [0.5 ** 2] * 3)
        self.meas_var = meas_std ** 2
        self.accel_std = accel_std          # process noise, as unmodelled accel

    @staticmethod
    def _F(dt: float) -> np.ndarray:
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        return F

    def _Q(self, dt: float) -> np.ndarray:
        """White-noise-acceleration process noise (Bar-Shalom's CV model)."""
        q = self.accel_std ** 2
        Q = np.zeros((6, 6))
        Q[:3, :3] = np.eye(3) * (dt ** 4 / 4.0) * q
        Q[:3, 3:] = np.eye(3) * (dt ** 3 / 2.0) * q
        Q[3:, :3] = Q[:3, 3:]
        Q[3:, 3:] = np.eye(3) * (dt ** 2) * q
        return Q

    def update(self, pos, dt: float) -> None:
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)
        H = np.zeros((3, 6))
        H[:, :3] = np.eye(3)
        S = H @ self.P @ H.T + np.eye(3) * self.meas_var
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (np.asarray(pos, float) - H @ self.x)
        self.P = (np.eye(6) - K @ H) @ self.P

    def forecast(self, horizons) -> tuple[np.ndarray, np.ndarray]:
        """Predicted centres and position sigma at each horizon."""
        centres, sigmas = [], []
        for h in np.atleast_1d(horizons):
            F = self._F(float(h))
            centres.append((F @ self.x)[:3])
            P = F @ self.P @ F.T + self._Q(float(h))
            sigmas.append(float(np.sqrt(np.trace(P[:3, :3]) / 3.0)))
        return np.asarray(centres), np.asarray(sigmas)

    def effective_radius(self, horizons, base_radius: float, n_sigma: float,
                         sigma_cap: float | None = None) -> np.ndarray:
        """Inflated radius, optionally capped by where the obstacle can be.

        The cap is not a fudge. A constant-velocity filter extrapolates without
        bound, so its 2 s covariance implies a sphere wider than the cell --
        measured at 2.40 m against a true mean error of 0.98 m. Inflating by
        that marks the whole workspace blocked and the robot stops dead. A cell
        knows the volume its obstacles are confined to, so the tube saturates
        there instead of growing forever.
        """
        sig = self.forecast(horizons)[1]
        if sigma_cap is not None:
            sig = np.minimum(sig, sigma_cap)
        return base_radius + n_sigma * sig


def time_to_collision(traj, times, t_now, tracker, *, base_radius: float,
                      n_sigma: float, clearance: float, horizon: float = 2.5,
                      sigma_cap: float | None = None):
    """Earliest horizon at which the arm enters the predicted tube.

    Returns (ttc, index, depth): ttc is None when the remaining path stays
    clear over `horizon`. `depth` is how far inside the tube the worst point
    gets, which is what the deformation below needs to know how hard to push.
    """
    future = [(i, t - t_now) for i, t in enumerate(times) if 0.0 <= t - t_now <= horizon]
    if not future:
        return None, None, 0.0
    idx = np.array([i for i, _ in future])
    hs = np.array([h for _, h in future])
    centres, sigmas = tracker.forecast(hs)
    if sigma_cap is not None:
        sigmas = np.minimum(sigmas, sigma_cap)
    radii = base_radius + n_sigma * sigmas + clearance
    for k, i in enumerate(idx):
        d = np.linalg.norm(arm_points(traj[i])[0] - centres[k], axis=1).min()
        if d < radii[k]:
            return float(hs[k]), int(i), float(radii[k] - d)
    return None, None, 0.0
