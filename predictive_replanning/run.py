"""Closed-loop MuJoCo runs, one row per (strategy, seed).

Everything is driven off the seed: the obstacle process, the measurement noise
the tracker sees, and nothing else. Two strategies on the same seed therefore
meet the *same* obstacle trajectory, which is the only way the difference
between them is attributable to the strategy rather than to luck. The
ME5250 report's results were explicitly "qualitative observations from
development testing rather than rigorous experimental trials"; these are
paired trials with a denominator.

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
from predictive_replanning.obstacle import ObstacleProcess
from predictive_replanning.predict import ObstacleTracker, arm_points, time_to_collision
from predictive_replanning.replan import deform_minimal, nominal_trajectory
from predictive_replanning.ur12e import ik

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


def _goals():
    """Pick and place targets in base_link, from the cell's own geometry."""
    top = PICK_TABLE["centre"][2] + PICK_TABLE["half"][2]
    pick = np.array([CUBE_XY[0][0], CUBE_XY[0][1], top + 0.10]) - np.array([0, 0, BASE_Z])
    place = np.array([PLACE_XY[0], PLACE_XY[1], 0.62]) - np.array([0, 0, BASE_Z])
    return pick, place


def run_one(strategy: str, seed: int, *, dt: float = 0.02, duration: float = 6.0,
            n_way: int = 60, base_radius: float = 0.09, n_sigma: float = 1.0,
            clearance: float = 0.02, ttc_threshold: float = 2.0,
            preempt_dist: float = 0.20, sigma_cap: float = 0.10,
            meas_std: float = 0.02, cooldown: float = 0.5,
            trace: list | None = None, frames: list | None = None,
            render_stride: int = 4, res: tuple[int, int] = (480, 640)) -> dict:
    model = mujoco.MjModel.from_xml_string(build_mjcf(obstacle_radius=base_radius))
    data = mujoco.MjData(model)
    adr = _arm_qpos_adr(model)
    mocap = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obstacle")]
    obst_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle_g")
    arm_geoms = [g for g in range(model.ngeom)
                 if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").endswith("_col")]

    def true_clearance() -> float:
        """Surface-to-surface, obstacle against every arm collision mesh."""
        return min(float(mujoco.mj_geomDistance(model, data, obst_geom, g, 2.0, None))
                   for g in arm_geoms)

    renderer = mujoco.Renderer(model, *res) if frames is not None else None
    if renderer is not None:
        # A free camera aimed at the workspace, not a fixed one in the MJCF:
        # lookat/azimuth/elevation is tunable in one place, and every GIF uses
        # the same values so two runs differ only by what the arm did.
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = (-0.55, 0.12, 0.48)
        cam.distance, cam.azimuth, cam.elevation = 1.95, 158.0, -16.0
    q_home = np.array([0.0, -1.2, 1.4, -1.6, -1.5708, 0.0])
    pick, place = _goals()
    q_start, ok_a, _ = ik(pick, q_home)
    q_goal, ok_b, _ = ik(place, q_start)
    if not (ok_a and ok_b):
        return dict(strategy=strategy, seed=seed, ik_failed=True)

    traj, times = nominal_trajectory(q_start, q_goal, n_way, duration)
    nominal = traj.copy()

    # Centre the obstacle's wander box on the middle of the nominal path, not
    # on a fixed point in the cell. The scenario under test is "something
    # enters the volume the arm is about to move through"; an obstacle parked
    # somewhere the arm never goes tests nothing, and with UR's real collision
    # meshes -- which are thinner than the capsules an earlier draft used --
    # the do-nothing baseline cleared 20 of 25 runs, leaving no headroom for
    # any strategy to show a difference. The box still spans a wide volume, so
    # the obstacle is not aimed at the arm, only placed where the work is.
    mid_tcp = arm_points(nominal[len(nominal) // 2])[0][6] + np.array([0.0, 0.0, BASE_Z])
    proc = ObstacleProcess(seed=seed, radius=base_radius,
                           centre=mid_tcp,
                           box_half=np.array([0.22, 0.30, 0.18]))
    rng = np.random.default_rng(seed + 5000)
    tracker = None

    for a, v in zip(adr, q_start):
        data.qpos[a] = v
    # Only the six arm actuators are commanded here; the two Hand-E finger
    # actuators keep their own targets, so ctrl is addressed by slice rather
    # than assigned wholesale.
    arm_act = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{j}")
                        for j in ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                                  "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")])
    data.ctrl[arm_act] = q_start
    mujoco.mj_forward(model, data)

    t = 0.0
    last_replan = -1e9
    replans, deviation, min_clear = 0, 0.0, np.inf
    collided = False
    steps = int(duration / dt) + 40

    for _step in range(steps):
        p = proc.step(dt)
        data.mocap_pos[mocap] = p
        obs = p + rng.normal(0.0, meas_std, 3)
        if tracker is None:
            tracker = ObstacleTracker(obs, meas_std=meas_std)
        else:
            tracker.update(obs, dt)

        i = int(np.clip(np.searchsorted(times, t), 0, n_way - 1))
        q_now = traj[i]

        # Surface distance on the state the simulator actually reached, not on
        # the commanded one: the controller lags, and scoring the command would
        # credit the planner for a pose the arm never held.
        p_rel = p - np.array([0.0, 0.0, BASE_Z])
        mujoco.mj_forward(model, data)
        d_true = true_clearance()
        min_clear = min(min_clear, d_true)
        if d_true <= 0.0:
            collided = True

        # what the reactive rule fires on: centre-to-skeleton, as before
        d_skel = float(np.linalg.norm(arm_points(q_now)[0] - p_rel, axis=1).min())

        if (strategy == "reactive" and d_skel < preempt_dist
                and i < n_way - 2 and t - last_replan >= cooldown):
            # Replan from here to the goal, obstacle taken where it is now.
            sub, sub_t = nominal_trajectory(q_now, q_goal, n_way - i, float(times[-1] - t))
            fixed, dev, _, _ = deform_minimal(
                sub, sub_t, 0.0, ObstacleTracker(p_rel, meas_std=meas_std),
                base_radius=base_radius, n_sigma=0.0, clearance=clearance,
                horizon=duration, sigma_cap=0.0, lock_before=0)
            traj = np.vstack([traj[:i], fixed])
            times = np.concatenate([times[:i], t + sub_t])
            deviation += dev
            replans += 1
            last_replan = t
        elif (strategy == "predictive" and i < n_way - 2
              and t - last_replan >= cooldown):
            ttc, _, _ = time_to_collision(
                traj, times, t, tracker, base_radius=base_radius,
                n_sigma=n_sigma, clearance=clearance, horizon=ttc_threshold,
                sigma_cap=sigma_cap)
            if ttc is not None:
                traj, dev, _, _ = deform_minimal(
                    traj, times, t, tracker, base_radius=base_radius,
                    n_sigma=n_sigma, clearance=clearance, horizon=ttc_threshold,
                    sigma_cap=sigma_cap, lock_before=i)
                deviation += dev
                replans += 1
                last_replan = t

        if trace is not None:
            trace.append(dict(t=t, q=traj[i].copy(), obstacle=p_rel.copy(),
                              clearance=d_true, replans=replans,
                              tcp=arm_points(traj[i])[0][6].copy()))
        data.ctrl[arm_act] = traj[i]
        mujoco.mj_step(model, data)
        if renderer is not None and len(frames) * render_stride <= _step:
            renderer.update_scene(data, camera=cam)
            frames.append(dict(px=renderer.render().copy(), t=t,
                               clearance=d_true, replans=replans,
                               hit=d_true <= 0.0))
        t += dt

    reached = float(np.linalg.norm(traj[-1] - q_goal))
    path_len = float(np.abs(np.diff(traj, axis=0)).sum())
    nominal_len = float(np.abs(np.diff(nominal, axis=0)).sum())
    return dict(strategy=strategy, seed=seed, collided=collided,
                min_clearance=round(min_clear, 4), replans=replans,
                deviation=round(deviation, 4), goal_error=round(reached, 6),
                path_len=round(path_len, 4), nominal_len=round(nominal_len, 4))


def main() -> None:
    _sig()
    ap = argparse.ArgumentParser(description="Predictive replanning trials in MuJoCo.")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--sweep-ttc", type=str, default="",
                    help="comma-separated TTC thresholds to sweep, e.g. 0.3,0.5,1.0,2.0")
    ap.add_argument("--ttc", type=float, default=2.0, help="TTC threshold, seconds")
    ap.add_argument("--n-sigma", type=float, default=1.0)
    ap.add_argument("--cooldown", type=float, default=0.5,
                    help="minimum seconds between replans; a real replan is not free")
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    if args.sweep_ttc:
        print(f"{'ttc':<8}{'no collision':<14}{'min clear':<12}{'replans':<10}{'path/nominal'}")
        base = [run_one("none", s) for s in range(args.trials)]
        safe = sum(not r["collided"] for r in base)
        print(f"{'none':<8}{safe}/{len(base):<12}"
              f"{np.mean([r['min_clearance'] for r in base]):<12.4f}{0.0:<10.1f}1.000")
        for ttc in [float(x) for x in args.sweep_ttc.split(",")]:
            g = [run_one("predictive", s, ttc_threshold=ttc, n_sigma=args.n_sigma,
                         cooldown=args.cooldown) for s in range(args.trials)]
            g = [r for r in g if not r.get("ik_failed")]
            safe = sum(not r["collided"] for r in g)
            print(f"{ttc:<8}{safe}/{len(g):<12}"
                  f"{np.mean([r['min_clearance'] for r in g]):<12.4f}"
                  f"{np.mean([r['replans'] for r in g]):<10.1f}"
                  f"{np.mean([r['path_len'] / r['nominal_len'] for r in g]):.3f}")
        return

    rows = [run_one(s, seed, ttc_threshold=args.ttc, n_sigma=args.n_sigma,
                    cooldown=args.cooldown)
            for s in STRATEGIES for seed in range(args.trials)]
    rows = [r for r in rows if not r.get("ik_failed")]

    print(f"{'strategy':<12}{'no collision':<14}{'min clear':<12}"
          f"{'replans':<10}{'deviation':<12}{'path/nominal'}")
    for s in STRATEGIES:
        g = [r for r in rows if r["strategy"] == s]
        if not g:
            continue
        safe = sum(not r["collided"] for r in g)
        print(f"{s:<12}{safe}/{len(g):<12}"
              f"{np.mean([r['min_clearance'] for r in g]):<12.4f}"
              f"{np.mean([r['replans'] for r in g]):<10.1f}"
              f"{np.mean([r['deviation'] for r in g]):<12.3f}"
              f"{np.mean([r['path_len'] / r['nominal_len'] for r in g]):.3f}")
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
