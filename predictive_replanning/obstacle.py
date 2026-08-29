"""The moving obstacle, and the randomness the robot is asked to predict.

Velocity follows an Ornstein-Uhlenbeck process rather than a random walk on
position. Two reasons, and both matter for what is being measured downstream:

  * OU velocity is mean-reverting, so the path stays smooth and bounded instead
    of wandering off or jittering. A hand reaching into a cell moves like this;
    Brownian position does not.
  * Its forward distribution is known in closed form. That means the predictor
    in predict.py can be checked against the truth it is trying to estimate --
    the growth of predicted uncertainty is a number with a right answer, not a
    tuning knob. A predictor scored only against its own residuals cannot tell
    a good model from a confident wrong one.

    dv = -theta (v - mu) dt + sigma dW

Over a horizon h, from a known v0:

    E[v(h)]   = mu + (v0 - mu) e^{-theta h}
    E[x(h)]   = x0 + mu h + (v0 - mu)(1 - e^{-theta h}) / theta
    Var[x(h)] = (sigma^2 / theta^2) [ h - 2(1 - e^{-theta h})/theta
                                        + (1 - e^{-2 theta h})/(2 theta) ]

The Var[x] expression is the standard integrated-OU variance. It is what makes
"predict the randomness" a statement with content: the robot avoids a tube that
widens at a rate the process actually has.
"""
from __future__ import annotations

__author__ = "".join(
    chr(c - 7) for c in (104, 105, 107, 124, 115, 39, 121, 104, 111, 116, 104, 117)
)

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ObstacleProcess:
    """A seeded OU-velocity obstacle confined to a box."""

    centre: np.ndarray = field(default_factory=lambda: np.array([-0.70, 0.10, 0.55]))
    box_half: np.ndarray = field(default_factory=lambda: np.array([0.26, 0.40, 0.22]))
    # For OU the per-axis velocity std is sigma / sqrt(2 theta), and theta sets
    # how fast direction is forgotten. Both are turned up here on purpose: at
    # theta 3.0 and sigma 0.9 the per-axis std is 0.367 m/s, an RMS 3-D speed of
    # 0.64 m/s, and the velocity autocorrelation time is 1/theta = 0.33 s.
    #
    # That is deliberately harder than a person reaching into a cell, which the
    # earlier setting (theta 1.4, sigma 0.28, 0.29 m/s) matched against the
    # 0.2-0.5 m/s band reported for collaborative workcells. The point of the
    # faster process is that direction is forgotten inside a third of a second,
    # so a constant-velocity forecast has almost nothing to hold on to and the
    # predictor is being asked to do something genuinely hard. Both numbers are
    # constructor arguments; pass the old pair to get the gentler obstacle back.
    theta: float = 3.0              # 1/s, how fast velocity reverts to mu
    sigma: float = 0.9              # m/s^(3/2), velocity noise
    mu: np.ndarray = field(default_factory=lambda: np.zeros(3))
    radius: float = 0.09
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.pos = self.centre.astype(float).copy()
        self.vel = self.rng.normal(0.0, 0.12, 3)

    def step(self, dt: float) -> np.ndarray:
        """Exact OU update for velocity, then integrate and reflect at the box."""
        e = np.exp(-self.theta * dt)
        # Stationary-increment form: exact for any dt, unlike Euler-Maruyama,
        # so the trajectory does not depend on the controller's step size.
        std = self.sigma * np.sqrt((1.0 - e ** 2) / (2.0 * self.theta))
        self.vel = self.mu + (self.vel - self.mu) * e + self.rng.normal(0.0, std, 3)
        self.pos = self.pos + self.vel * dt

        # Reflect rather than clamp: clamping parks the obstacle on a face and
        # silently kills the velocity the predictor is trying to track.
        lo, hi = self.centre - self.box_half, self.centre + self.box_half
        for i in range(3):
            if self.pos[i] < lo[i]:
                self.pos[i] = lo[i] + (lo[i] - self.pos[i])
                self.vel[i] = abs(self.vel[i])
            elif self.pos[i] > hi[i]:
                self.pos[i] = hi[i] - (self.pos[i] - hi[i])
                self.vel[i] = -abs(self.vel[i])
        return self.pos.copy()

    def true_forecast(self, horizons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The process's own forward mean and position std, ignoring the box.

        This is the truth the predictor is measured against. Reflection is not
        modelled here, so it is only valid away from the walls -- which is
        exactly where it is used as a reference.
        """
        h = np.asarray(horizons, dtype=float)[:, None]
        e = np.exp(-self.theta * h)
        mean = self.pos + self.mu * h + (self.vel - self.mu) * (1.0 - e) / self.theta
        var = (self.sigma ** 2 / self.theta ** 2) * (
            h - 2.0 * (1.0 - e) / self.theta + (1.0 - np.exp(-2.0 * self.theta * h)) / (2.0 * self.theta)
        )
        return mean, np.sqrt(np.maximum(var, 0.0))
