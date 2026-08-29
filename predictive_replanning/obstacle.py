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
    """An OU-velocity obstacle confined to a box.

    Takes a generator rather than a seed. Callers that want a reproducible
    motion pass a seeded generator; callers that want a real random draw pass
    nothing. Owning a seed here made every consumer inherit reproducibility it
    had not asked for, and made it look like the experiment depended on one.
    """

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
    rng: np.random.Generator | None = None

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = np.random.default_rng()
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


@dataclass
class Track:
    """One recorded obstacle motion, and the observations of it.

    Seeds were the wrong interface. What a paired comparison needs is that
    every strategy meets the *same obstacle motion*, and a seed only delivers
    that as a side effect of nothing else touching the generator in between --
    which is a property of the whole program, not of the experiment. Two
    strategies that draw a different number of random numbers silently diverge,
    and the tell is a comparison that looks fine and means nothing.

    A track is the motion itself. It is drawn once from the OS entropy pool, so
    it is genuinely random rather than a fixed list of integers, and then
    replayed. Measurement noise is recorded with it, so the two strategies also
    see identical observations and the only difference left between two runs is
    the strategy.
    """

    positions: np.ndarray            # (N, 3) where the obstacle really was
    observations: np.ndarray         # (N, 3) what the tracker was shown
    dt: float
    radius: float
    theta: float
    sigma: float

    def __len__(self) -> int:
        return len(self.positions)

    def at(self, step: int) -> tuple[np.ndarray, np.ndarray]:
        k = min(step, len(self.positions) - 1)
        return self.positions[k], self.observations[k]

    @property
    def rms_speed(self) -> float:
        v = np.diff(self.positions, axis=0) / self.dt
        return float(np.sqrt(np.mean(np.sum(v ** 2, axis=1))))


def record_track(*, steps: int, dt: float, centre, box_half, radius: float,
                 theta: float, sigma: float, meas_std: float,
                 rng: np.random.Generator | None = None) -> Track:
    """Draw one obstacle motion. Entropy from the OS unless a generator is given."""
    rng = rng or np.random.default_rng()
    # One generator for the whole batch, handed down rather than reseeded.
    proc = ObstacleProcess(centre=np.asarray(centre, float),
                           box_half=np.asarray(box_half, float),
                           theta=theta, sigma=sigma, radius=radius, rng=rng)
    pos = np.empty((steps, 3))
    for k in range(steps):
        pos[k] = proc.step(dt)
    obs = pos + rng.normal(0.0, meas_std, pos.shape)
    return Track(positions=pos, observations=obs, dt=dt, radius=radius,
                 theta=theta, sigma=sigma)


def record_tracks(n: int, **kw) -> list[Track]:
    """A batch of independent motions, all from one entropy draw."""
    rng = np.random.default_rng()
    return [record_track(rng=rng, **kw) for _ in range(n)]


def save_tracks(tracks: list[Track], path) -> None:
    """Write a batch to disk.

    This is how a number in the README stays checkable without a seed. The
    motions themselves are the record; anyone can replay the exact obstacles a
    result was measured against, which a seed only promises as long as nothing
    else in the program draws from the same generator.
    """
    np.savez_compressed(
        path,
        positions=np.stack([t.positions for t in tracks]),
        observations=np.stack([t.observations for t in tracks]),
        meta=np.array([[t.dt, t.radius, t.theta, t.sigma] for t in tracks]))


def load_tracks(path) -> list[Track]:
    z = np.load(path)
    return [Track(positions=p, observations=o, dt=float(m[0]), radius=float(m[1]),
                  theta=float(m[2]), sigma=float(m[3]))
            for p, o, m in zip(z["positions"], z["observations"], z["meta"])]


#: Velocity-tracking rate while closing. Higher than the loiter value so the
#: obstacle achieves its commanded approach speed instead of lagging it.
PURSUIT_THETA = 8.0
#: Noise during the dart, as a fraction of the loiter noise. Enough that
#: the approach is not a straight line the tracker could extrapolate in
#: closed form, little enough that the obstacle reaches what it aimed at.
PURSUIT_NOISE = 0.45


def record_intercept_track(*, tcp_path, times, carry_mask, dt: int, steps: int,
                           radius: float, theta: float, sigma: float,
                           meas_std: float, rng: np.random.Generator | None = None,
                           standoff=(0.30, 0.50), approach=(1.2, 2.0)) -> Track:
    """An obstacle that goes for the arm on purpose.

    A random walk in a box only wanders into the path some of the time, so most
    trials never posed a conflict at all and the comparison measured how often
    the obstacle happened to be elsewhere. Worse, it gave the predictor nothing
    worth predicting: a mean-reverting wander has no intent to estimate, so the
    filter's velocity carried almost no information about where the obstacle
    would be when it mattered.

    This one picks a waypoint on the arm's own planned path, works out when the
    arm will be there, and sets off from a standoff distance so as to arrive at
    the same place at the same time. Every trial is therefore a real conflict,
    and the obstacle has a genuine closing velocity -- which is exactly the
    quantity a constant-velocity filter exists to estimate.

    It waits before it moves. Setting off at t=0 meant creeping in at 0.08 m/s
    over six seconds, and integrated OU noise over that long is several times
    the standoff -- the obstacle drifted off its own aim and arrived 0.09 to
    0.16 m wide, so half the trials still posed no conflict. Loitering and then
    closing over a second or two is both what a hand does and short enough that
    the noise cannot swamp the approach.

    It is still random in the ways that matter: which waypoint it aims at, which
    direction it comes from, how far it starts, how long it takes to close, and
    OU noise the whole way, so it neither travels in a straight line nor arrives
    exactly where it aimed. What is no longer random is whether there is
    anything to avoid.
    """
    rng = rng or np.random.default_rng()
    tcp_path = np.asarray(tcp_path, float)
    times = np.asarray(times, float)
    carry = np.where(np.asarray(carry_mask, bool))[0]

    # Aim at a point in the middle of the carry, away from the very ends where
    # the arm is still leaving the bench or already over the target.
    lo, hi = int(0.20 * len(carry)), int(0.85 * len(carry))
    k = int(carry[rng.integers(lo, max(lo + 1, hi))])
    aim = tcp_path[k]
    t_hit = float(times[k])
    _ = aim              # kept for readability; the pursuit re-aims each step

    # Come in from a random direction, mostly horizontal: a hand reaches across
    # a cell, it does not drop out of the ceiling.
    u = rng.normal(size=3)
    u[2] *= 0.35
    u /= np.linalg.norm(u)
    start = aim + u * float(rng.uniform(*standoff))

    t_close = float(rng.uniform(*approach))
    t_go = max(0.0, t_hit - t_close)

    proc = ObstacleProcess(centre=start, box_half=np.array([2.0, 2.0, 2.0]),
                           theta=theta, sigma=sigma * 0.25, radius=radius, rng=rng)
    proc.pos = start.copy()
    proc.mu = np.zeros(3)
    proc.vel = np.zeros(3)

    pos = np.empty((steps, 3))
    closing = False
    for s in range(steps):
        t = s * dt
        if t >= t_go and t <= t_hit:
            # Pursue, rather than fire once at a predicted point. A single
            # aimed shot missed by 0.11 to 0.16 m once OU noise and the arm's
            # own motion were added, so nearly half the trials posed no
            # conflict. Re-aiming each step at where the arm goes if it does
            # NOT replan makes the baseline a collision by construction: doing
            # nothing gets hit, and only deviating from that path escapes,
            # which is the thing under test.
            #
            # The obstacle is chasing a recorded path, not the live arm. It
            # cannot react to a replan, so a robot that moves early genuinely
            # gets away -- it is not being cheated by an omniscient pursuer.
            if not closing:
                closing = True
                proc.sigma = sigma * PURSUIT_NOISE
                # Make the velocity actually follow the command while closing.
                # At theta 1.4 the velocity time constant is 0.71 s, comparable
                # to the dart itself, so the obstacle spent the whole approach
                # accelerating and arrived late and wide. A shorter constant
                # lets it reach the closing speed it was told to hold; the
                # noise is unchanged, so the path is no straighter.
                proc.theta = PURSUIT_THETA
            here = tcp_path[min(int(t / dt), len(tcp_path) - 1)]
            proc.mu = (here - proc.pos) / max(t_hit - t, 3.0 * dt)
        elif closing and t > t_hit:
            proc.mu = np.zeros(3)          # past it; let it drift away
        pos[s] = proc.step(dt)
    obs = pos + rng.normal(0.0, meas_std, pos.shape)
    return Track(positions=pos, observations=obs, dt=dt, radius=radius,
                 theta=theta, sigma=sigma)


def record_intercept_tracks(n: int, **kw) -> list[Track]:
    rng = np.random.default_rng()
    return [record_intercept_track(rng=rng, **kw) for _ in range(n)]
