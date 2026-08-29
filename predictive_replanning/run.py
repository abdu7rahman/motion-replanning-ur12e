"""Closed-loop MuJoCo runs, one row per (strategy, obstacle track).

Every strategy is replayed against the *same recorded obstacle motion*, drawn
from the OS entropy pool rather than reproduced from a seed. A seed only
delivers a paired comparison as a side effect of nothing else touching the
generator in between, which is a property of the whole program rather than of
the experiment -- two strategies that draw a different number of random numbers
diverge silently, and the tell is a comparison that looks fine and means
nothing. Recording the motion, measurement noise included, makes the pairing
the thing it claims to be.

The ME5250 report's results were explicitly "qualitative observations from
development testing rather than rigorous experimental trials"; these are paired
trials with a denominator.

A run succeeds when the cube is on the place table. Avoidance alone is not
success -- an arm that freezes also never collides, and a run that dodges
perfectly and drops the cube has failed at the job it was given.

Two distances are computed and they are not the same thing.

What is *reported* is `mj_geomDistance` between the obstacle and every one of
the arm's collision meshes plus the Hand-E body -- true surface-to-surface, on
UR's own hulls, measured on the state the simulator actually reached. Zero
means touching, so a collision is distance <= 0.

What the *planner* uses is the point skeleton in predict.py: joint origins and
samples down each shaft. That is a deliberate approximation and a cheap one,
because the predictive check runs it over every future waypoint on every step.
Keeping the two apart means the planner is never scored on its own
simplification -- if the skeleton were the whole story, an arm could clear the
skeleton and still put a forearm through the obstacle, and nothing would say so.

Contact is not used: the obstacle is a mocap body with contype=0, because an
obstacle the arm can shove aside is not the obstacle under study.
"""
from __future__ import annotations

__author__ = "".join(
    chr(c - 7) for c in (104, 105, 107, 124, 115, 39, 121, 104, 111, 116, 104, 117)
)

import argparse
import json

import mujoco
import numpy as np

from predictive_replanning.cell import PICK_TABLE, PLACE_XY, build_mjcf, CUBE_XY
from predictive_replanning.obstacle import Track, load_tracks, record_tracks, save_tracks
from predictive_replanning.predict import ObstacleTracker, arm_points, time_to_collision
from predictive_replanning.replan import deform_minimal, soft_mask
from predictive_replanning.task import pick_and_place, placed, solve
from predictive_replanning.ur12e import fk_tcp_pos

BASE_Z = 0.18                       # base_link height in the MJCF
STRATEGIES = ("none", "reactive", "predictive")


def _arm_qpos_adr(model):
    """Address joints by name. The cube freejoints are declared first, so the
    arm does not start at qpos[0] -- assuming it does silently reads cube
    state as joint angles."""
    names = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
             "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
    return np.array([model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names])




#: The compiled model, keyed on obstacle radius. Building it parses 55 meshes,
#: which at a hundred-odd trials dominates the run and was enough to get a
#: batch killed for memory. The model is immutable across trials -- only MjData
#: changes -- so it is compiled once and reset per run.
_MODEL_CACHE: dict[float, "mujoco.MjModel"] = {}


def _model_for(radius: float):
    if radius not in _MODEL_CACHE:
        _MODEL_CACHE[radius] = mujoco.MjModel.from_xml_string(
            build_mjcf(obstacle_radius=radius))
    return _MODEL_CACHE[radius]


def carry_centre() -> tuple[np.ndarray, float, int]:
    """The middle of the pick-to-place corridor, and how long a run lasts.

    This is the halfway point of the *transfer* specifically -- the straight
    carry from above the cube to above the place table -- taken at the tool
    centre point. Two earlier versions of this were wrong in ways that flattered
    the results:

      * it centred on the wrist origin rather than the TCP, so the hazard sat
        15 cm behind the part of the arm carrying the cube;
      * it averaged over every deformable waypoint, approach included, which
        pulls the centre back toward the pick table and away from the carry.

    Both put the obstacle off to one side of the work. Combined with a box wide
    enough to manufacture headroom, "avoided" mostly meant "was never a threat",
    and the comparison measured distance rather than replanning.
    """
    q_home = np.array([0.0, -1.2, 1.4, -1.6, -1.5708, 0.0])
    phases = pick_and_place()
    traj, times, _, _, _ = solve(phases, q_home)
    names = [p.name for p in phases]
    start = fk_tcp_pos(traj[(names.index("lift") + 1) * 14 - 1])
    end = fk_tcp_pos(traj[(names.index("transfer") + 1) * 14 - 1])
    mid = 0.5 * (start + end) + np.array([0.0, 0.0, BASE_Z])
    return mid, float(times[-1]), len(traj)


def make_tracks(n: int, *, dt: float = 0.02, radius: float = 0.07,
                theta: float = 1.4, sigma: float = 0.28,
                meas_std: float = 0.02) -> list[Track]:
    """A batch of obstacle motions for a comparison. No seeds: drawn from the
    OS entropy pool, then replayed identically for every strategy."""
    mid, duration, _ = carry_centre()
    # Tight around the corridor: the obstacle stays in the way rather than
    # wandering somewhere the arm never goes. Narrow across the path (x, z) and
    # free to slide along it (y), which is what a hand reaching into a transfer
    # actually does.
    return record_tracks(n, steps=int(duration / dt) + 60, dt=dt, centre=mid,
                         box_half=(0.12, 0.22, 0.12), radius=radius,
                         theta=theta, sigma=sigma, meas_std=meas_std)


def run_one(strategy: str, track: Track, *, dt: float = 0.02,
            n_sigma: float = 1.0, clearance: float = 0.02, ttc_threshold: float = 2.0,
            preempt_dist: float = 0.20, sigma_cap: float = 0.10,
            meas_std: float = 0.02, cooldown: float = 0.5,
            trace: list | None = None, frames: list | None = None,
            render_stride: int = 4, res: tuple[int, int] = (480, 640)) -> dict:
    base_radius = track.radius
    model = _model_for(base_radius)
    data = mujoco.MjData(model)
    adr = _arm_qpos_adr(model)
    nid = lambda t, n: mujoco.mj_name2id(model, t, n)
    mocap = model.body_mocapid[nid(mujoco.mjtObj.mjOBJ_BODY, "obstacle")]
    obst_geom = nid(mujoco.mjtObj.mjOBJ_GEOM, "obstacle_g")
    cube_bid = nid(mujoco.mjtObj.mjOBJ_BODY, "cube_0")
    arm_geoms = [g for g in range(model.ngeom)
                 if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").endswith("_col")]
    env_geoms = [nid(mujoco.mjtObj.mjOBJ_GEOM, n)
                 for n in ("floor", "pedestal", "pick_table", "place_table")]
    arm_act = np.array([nid(mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{j}")
                        for j in ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                                  "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")])
    grip_act = np.array([nid(mujoco.mjtObj.mjOBJ_ACTUATOR, "act_grip_l"),
                         nid(mujoco.mjtObj.mjOBJ_ACTUATOR, "act_grip_r")])
    renderer = mujoco.Renderer(model, *res) if frames is not None else None
    if renderer is not None:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = (-0.52, 0.16, 0.44)
        cam.distance, cam.azimuth, cam.elevation = 1.85, 108.0, -22.0

    q_home = np.array([0.0, -1.2, 1.4, -1.6, -1.5708, 0.0])
    try:
        traj, times, grip, deformable, holding = solve(pick_and_place(), q_home)
    except RuntimeError as exc:
        return dict(strategy=strategy, ik_failed=True, why=str(exc))
    nominal = traj.copy()
    n_way = len(traj)
    duration = float(times[-1])

    def true_clearance() -> float:
        """Surface-to-surface, obstacle against every arm collision mesh."""
        return min(float(mujoco.mj_geomDistance(model, data, obst_geom, g, 2.0, None))
                   for g in arm_geoms)

    def env_penetration() -> float:
        """Deepest arm-into-furniture overlap right now, 0 when clear.

        Separate from the obstacle: hitting the bench is a planning fault, not a
        dodge that failed, and lumping them together hides which one happened.
        The cube is excluded -- the gripper is supposed to touch it.
        """
        worst = 0.0
        for c in range(data.ncon):
            g1, g2 = data.contact[c].geom1, data.contact[c].geom2
            pair = {g1, g2}
            if pair & set(arm_geoms) and pair & set(env_geoms):
                worst = min(worst, float(data.contact[c].dist))
        return -worst

    allow = soft_mask(deformable)
    tracker = None

    for a, v in zip(adr, traj[0]):
        data.qpos[a] = v
    data.ctrl[arm_act] = traj[0]
    data.ctrl[grip_act] = grip[0]
    mujoco.mj_forward(model, data)

    t, last_replan = 0.0, -1e9
    replans, deviation, min_clear = 0, 0.0, np.inf
    hit_obstacle = False
    worst_env = 0.0

    for _step in range(int(duration / dt) + 60):
        p, obs = track.at(_step)
        data.mocap_pos[mocap] = p
        tracker = (ObstacleTracker(obs, meas_std=meas_std) if tracker is None
                   else (tracker.update(obs, dt) or tracker))

        i = int(np.clip(np.searchsorted(times, t), 0, n_way - 1))
        q_now = traj[i]

        p_rel = p - np.array([0.0, 0.0, BASE_Z])
        mujoco.mj_forward(model, data)
        d_true = true_clearance()
        min_clear = min(min_clear, d_true)
        if d_true <= 0.0:
            hit_obstacle = True
        worst_env = max(worst_env, env_penetration())

        d_skel = float(np.linalg.norm(arm_points(q_now)[0] - p_rel, axis=1).min())

        if deformable[i] and t - last_replan >= cooldown and i < n_way - 2:
            if strategy == "reactive" and d_skel < preempt_dist:
                traj, dev, _, _ = deform_minimal(
                    traj, times, t, ObstacleTracker(p_rel, meas_std=meas_std),
                    base_radius=base_radius, n_sigma=0.0, clearance=clearance,
                    horizon=duration, sigma_cap=0.0, lock_before=i,
                    allow=allow)
                deviation += dev
                replans += 1
                last_replan = t
            elif strategy == "predictive":
                ttc, _, _ = time_to_collision(
                    traj, times, t, tracker, base_radius=base_radius,
                    n_sigma=n_sigma, clearance=clearance, horizon=ttc_threshold,
                    sigma_cap=sigma_cap)
                if ttc is not None:
                    traj, dev, _, _ = deform_minimal(
                        traj, times, t, tracker, base_radius=base_radius,
                        n_sigma=n_sigma, clearance=clearance, horizon=ttc_threshold,
                        sigma_cap=sigma_cap, lock_before=i, allow=allow)
                    deviation += dev
                    replans += 1
                    last_replan = t


        data.ctrl[arm_act] = traj[i]
        data.ctrl[grip_act] = grip[i]
        mujoco.mj_step(model, data)

        if trace is not None:
            trace.append(dict(t=t, i=i, q=traj[i].copy(), obstacle=p_rel.copy(),
                              clearance=d_true, replans=replans,
                              cube_z=float(data.xpos[cube_bid][2]),
                              track_err=float(np.linalg.norm(data.qpos[adr] - traj[i]))))
        if renderer is not None and len(frames) * render_stride <= _step:
            renderer.update_scene(data, camera=cam)
            frames.append(dict(px=renderer.render().copy(), t=t, clearance=d_true,
                               replans=replans, hit=d_true <= 0.0))
        t += dt

    # let the cube settle before scoring where it ended up
    for _ in range(150):
        mujoco.mj_step(model, data)
    # World frame, not base_link. The tables are placed in the worldbody, so
    # placed() compares against world coordinates; subtracting the base height
    # here scored a cube sitting exactly on target as 0.18 m too low.
    ok, xy_err, dz = placed(data.xpos[cube_bid])
    return dict(strategy=strategy,
                success=bool(ok and not hit_obstacle and worst_env < 0.005),
                placed=bool(ok), hit_obstacle=hit_obstacle,
                env_penetration=round(worst_env, 5),
                place_xy_err=round(xy_err, 4), place_dz=round(dz, 4),
                min_clearance=round(min_clear, 4), replans=replans,
                deviation=round(deviation, 4),
                path_len=round(float(np.abs(np.diff(traj, axis=0)).sum()), 4),
                nominal_len=round(float(np.abs(np.diff(nominal, axis=0)).sum()), 4))


def _row(label: str, g: list[dict]) -> str:
    """One summary line. Success is placed AND untouched; the two columns
    beside it are there so a strategy that trades one for the other is
    visible rather than averaged away."""
    n = len(g)
    return (f"{label:<12}{sum(r['success'] for r in g)}/{n:<8}"
            f"{sum(r['placed'] for r in g)}/{n:<7}"
            f"{sum(not r['hit_obstacle'] for r in g)}/{n:<7}"
            f"{np.mean([r['min_clearance'] for r in g]):<12.4f}"
            f"{np.mean([r['replans'] for r in g]):<10.1f}"
            f"{np.mean([r['path_len'] / r['nominal_len'] for r in g]):.3f}")


def main() -> None:
    _sig()
    ap = argparse.ArgumentParser(description="Predictive replanning trials in MuJoCo.")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--sweep-ttc", type=str, default="",
                    help="comma-separated TTC thresholds to sweep, e.g. 0.3,0.5,1.0,2.0")
    ap.add_argument("--ttc", type=float, default=2.0, help="TTC threshold, seconds")
    ap.add_argument("--n-sigma", type=float, default=1.0)
    ap.add_argument("--obstacle-sigma", type=float, default=0.28,
                    help="OU velocity noise; higher is more erratic")
    ap.add_argument("--obstacle-theta", type=float, default=1.4,
                    help="OU reversion rate; higher forgets direction faster")
    ap.add_argument("--cooldown", type=float, default=0.5,
                    help="minimum seconds between replans; a real replan is not free")
    ap.add_argument("--tracks", type=str, default="",
                    help="replay obstacle motions from this .npz instead of drawing new ones")
    ap.add_argument("--save-tracks", type=str, default="",
                    help="write the drawn motions here so the run can be repeated exactly")
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    if args.tracks:
        tracks = load_tracks(args.tracks)[:args.trials]
    else:
        tracks = make_tracks(args.trials, theta=args.obstacle_theta,
                             sigma=args.obstacle_sigma)
    if args.save_tracks:
        save_tracks(tracks, args.save_tracks)
    print(f"{len(tracks)} obstacle tracks, RMS speed "
          f"{np.mean([t.rms_speed for t in tracks]):.3f} m/s "
          f"(theta {tracks[0].theta}, sigma {tracks[0].sigma})"
          + (f", replayed from {args.tracks}" if args.tracks else "") + "\n")

    if args.sweep_ttc:
        print(f"{'ttc':<12}{'success':<10}{'placed':<9}{'no hit':<9}"
              f"{'min clear':<12}{'replans':<10}{'path/nominal'}")
        print(_row("none", [run_one("none", tk) for tk in tracks]))
        for ttc in [float(x) for x in args.sweep_ttc.split(",")]:
            g = [run_one("predictive", tk, ttc_threshold=ttc, n_sigma=args.n_sigma,
                         cooldown=args.cooldown) for tk in tracks]
            print(_row(f"{ttc}", [r for r in g if not r.get("ik_failed")]))
        return

    rows = [run_one(s, tk, ttc_threshold=args.ttc, n_sigma=args.n_sigma,
                    cooldown=args.cooldown)
            for s in STRATEGIES for tk in tracks]
    rows = [r for r in rows if not r.get("ik_failed")]

    print(f"{'strategy':<12}{'success':<10}{'placed':<9}{'no hit':<9}"
          f"{'min clear':<12}{'replans':<10}{'path/nominal'}")
    for s in STRATEGIES:
        g = [r for r in rows if r["strategy"] == s]
        if g:
            print(_row(s, g))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)
        print(f"\nwrote {args.json}")


def _sig():
    """Author signature. stderr, tty-only, so redirected output stays clean."""
    import os, sys
    if os.environ.get("NO_BANNER") == "1" or not sys.stderr.isatty():
        return
    print("  " + "".join(chr(c - 7) for c in
          (104,105,107,124,115,39,121,104,111,116,104,117)), file=sys.stderr)


if __name__ == "__main__":
    main()
