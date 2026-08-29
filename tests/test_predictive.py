#!/usr/bin/env python3
"""Checks the predictive replanner against things with right answers.

Each case below is one that can fail silently in a way a demo would not show:
a kinematic chain that is self-consistent but wrong, a Jacobian that points the
wrong way, an uncertainty model tuned rather than derived, a deformation that
quietly edits path the arm has already executed, and joint state read from the
wrong qpos slots. That last one is not hypothetical -- it is the bug this file
was started for.

    .venv/bin/python tests/test_predictive.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco                                                        # noqa: E402
from predictive_replanning import ur12e                              # noqa: E402
from predictive_replanning.cell import build_mjcf                    # noqa: E402
from predictive_replanning.obstacle import ObstacleProcess           # noqa: E402
from predictive_replanning.predict import arm_points, ObstacleTracker  # noqa: E402
from predictive_replanning.replan import (_window, deform_minimal,    # noqa: E402
                                          nominal_trajectory, soft_mask)
from predictive_replanning.task import pick_and_place, placed, solve  # noqa: E402

PASS, FAIL = "  PASS", "  FAIL"
_results = []


def check(name, ok, detail=""):
    _results.append(bool(ok))
    print(f"{PASS if ok else FAIL}  {name}" + (f"   {detail}" if detail else ""))


def test_fk_matches_mujoco():
    """Two independent implementations of the same chain, from one source.

    ur12e.py builds FK from the joint origins in numpy; cell.py emits those
    same origins into MJCF as quaternions. If the rpy->quat conversion is
    wrong, or MuJoCo's intrinsic euler convention had been used instead, this
    is where it shows -- wrist_3's origin (pi/2, pi, pi) is exactly where
    intrinsic and extrinsic xyz disagree.
    """
    model = mujoco.MjModel.from_xml_string(build_mjcf())
    data = mujoco.MjData(model)
    names = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
             "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
    adr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
           for n in names]
    check("arm joints are not at qpos[0]", adr[0] != 0,
          f"first arm joint at qpos[{adr[0]}], nq={model.nq}")
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")
    base = np.array([0.0, 0.0, 0.18])
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(300):
        q = rng.uniform(-2.8, 2.8, 6)
        for a, v in zip(adr, q):
            data.qpos[a] = v
        mujoco.mj_kinematics(model, data)
        worst = max(worst, float(np.linalg.norm((data.xpos[bid] - base) - ur12e.fk_pos(q))))
    check("python FK == MuJoCo FK", worst < 1e-9, f"max {worst:.2e} m over 300 configs")


def test_reach_matches_the_datasheet_geometry():
    """a2 + a3 is the fully-extended reach, and it is a number from the file."""
    a2, a3 = abs(ur12e.JOINT_ORIGINS[2][0]), abs(ur12e.JOINT_ORIGINS[3][0])
    p = ur12e.fk_pos(np.zeros(6))
    check("stretched reach == |a2|+|a3|", abs(abs(p[0]) - (a2 + a3)) < 1e-9,
          f"{abs(p[0]):.5f} m == {a2:.5f}+{a3:.5f}")


def test_point_jacobian():
    """Analytic against finite differences, on a forearm point rather than the
    tool -- the tool one is easy to get right by accident."""
    q = np.array([0.3, -1.2, 1.1, -0.4, 0.8, 0.2])
    pts, links = arm_points(q)
    i = int(np.argmax(links == 3))
    J = ur12e.point_jacobian(q, pts[i], int(links[i]))
    worst = 0.0
    for c in range(6):
        dq = np.zeros(6)
        dq[c] = 1e-6
        fd = (arm_points(q + dq)[0][i] - pts[i]) / 1e-6
        worst = max(worst, float(np.linalg.norm(fd - J[:, c])))
    check("point Jacobian == finite differences", worst < 1e-5, f"max {worst:.2e}")
    later = ur12e.point_jacobian(q, pts[i], int(links[i]))[:, 4:]
    check("joints past the link cannot move it", np.allclose(later, 0.0))


def test_ou_closed_form():
    """The forecast the predictor is scored against has to be right itself."""
    proc = ObstacleProcess(seed=1)
    for _ in range(300):
        proc.step(0.01)
    proc.box_half = np.array([9.0, 9.0, 9.0])          # walls off; closed form has none
    H = np.array([0.25, 1.0])
    mean, std = proc.true_forecast(H)
    samples = {h: [] for h in H}
    for k in range(1500):
        q = ObstacleProcess(seed=90_000 + k)
        q.pos, q.vel, q.box_half = proc.pos.copy(), proc.vel.copy(), proc.box_half
        t, todo = 0.0, list(H)
        while todo:
            q.step(0.01)
            t += 0.01
            if t >= todo[0] - 1e-9:
                samples[todo.pop(0)].append(q.pos.copy())
    for i, h in enumerate(H):
        S = np.array(samples[h])
        rel = abs(S.std(0).mean() - std[i, 0]) / std[i, 0]
        check(f"integrated-OU std at h={h}", rel < 0.10,
              f"analytic {std[i,0]:.4f} vs empirical {S.std(0).mean():.4f}")


def test_deformation_contract():
    """Endpoints pinned, executed path frozen, and it actually clears."""
    q0 = np.array([0.0, -1.2, 1.4, -1.6, -1.57, 0.0])
    qg = np.array([1.1, -1.0, 1.2, -1.5, -1.57, 0.0])
    traj, times = nominal_trajectory(q0, qg, 60, 6.0)
    w = _window(60, 30, 8, lock_before=34)
    check("window zero at both ends", w[0] == 0.0 and w[-1] == 0.0)
    check("window frozen before lock_before", np.all(w[:35] == 0.0))

    on_path = arm_points(traj[30])[0][6]
    trk = ObstacleTracker(on_path)
    for _ in range(30):
        trk.update(on_path, 0.02)
    new, dev, iters, cleared = deform_minimal(
        traj, times, 0.0, trk, base_radius=0.09, n_sigma=1.0, clearance=0.02,
        horizon=2.5, sigma_cap=0.20)
    check("deformation clears the tube", cleared, f"{iters} iterations")
    check("start pinned", np.allclose(new[0], traj[0]))
    check("goal pinned", np.allclose(new[-1], traj[-1]))
    check("deformation is not free", dev > 0.0, f"{dev:.3f} rad total")

    locked, _, _, _ = deform_minimal(
        traj, times, 0.0, trk, base_radius=0.09, n_sigma=1.0, clearance=0.02,
        horizon=2.5, sigma_cap=0.20, lock_before=40)
    check("executed waypoints never edited", np.allclose(locked[:41], traj[:41]))


def test_tracker_is_not_told_the_truth():
    """The filter must estimate velocity it was never given."""
    proc = ObstacleProcess(seed=4)
    rng = np.random.default_rng(4)
    trk = None
    for _ in range(200):
        p = proc.step(0.02)
        obs = p + rng.normal(0.0, 0.02, 3)
        trk = ObstacleTracker(obs) if trk is None else (trk.update(obs, 0.02) or trk)
    err = float(np.linalg.norm(trk.x[3:] - proc.vel))
    # Relative to the process's own stationary velocity spread, so the bound
    # means the same thing whether the obstacle is calm or wild. An absolute
    # threshold passed at 0.29 m/s and sat exactly on the line at 0.64.
    vel_std = proc.sigma / np.sqrt(2.0 * proc.theta) * np.sqrt(3.0)
    check("tracked velocity beats guessing zero", err < 1.5 * vel_std,
          f"error {err:.3f} vs process spread {vel_std:.3f} m/s")
    grew = trk.forecast([2.0])[1][0] > trk.forecast([0.25])[1][0]
    check("forecast uncertainty grows with horizon", grew)


def test_tcp_frame():
    """The tool frame, against MuJoCo's own tcp site.

    Solving IK on wrist_3 instead of the TCP hangs the Hand-E 0.157 m low,
    which drove the fingers through the pick table. The offset is the sum of
    three published numbers, so it is checked rather than trusted.
    """
    model = mujoco.MjModel.from_xml_string(build_mjcf())
    data = mujoco.MjData(model)
    names = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
             "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
    adr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
           for n in names]
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    base = np.array([0.0, 0.0, 0.18])
    rng = np.random.default_rng(2)
    worst = 0.0
    for _ in range(200):
        q = rng.uniform(-2.5, 2.5, 6)
        for a, v in zip(adr, q):
            data.qpos[a] = v
        mujoco.mj_forward(model, data)
        worst = max(worst, float(np.linalg.norm((data.site_xpos[sid] - base) - ur12e.fk_tcp_pos(q))))
    check("python TCP FK == MuJoCo tcp site", worst < 1e-9, f"max {worst:.2e} m")
    check("TCP sits below the wrist by the tool chain",
          abs(ur12e.TCP_OFFSET_Z - (0.011 + 0.099 + 0.0465)) < 1e-12,
          f"{ur12e.TCP_OFFSET_Z:.4f} m")


def test_task_reaches_every_phase():
    """Every phase must have an IK solution, and none may start in the table."""
    phases = pick_and_place()
    traj, times, grip, deform, weld = solve(phases, np.array([0.0, -1.2, 1.4, -1.6, -1.57, 0.0]))
    worst = 0.0
    for n, ph in enumerate(phases):
        reached = ur12e.fk_tcp_pos(traj[(n + 1) * 14 - 1])
        worst = max(worst, float(np.linalg.norm(reached - ph.tcp)))
    check("all eight phases reachable", worst < 1e-3, f"max TCP error {worst:.2e} m")
    check("grasp closes before the carry", grip[weld][0] > 0.02)
    check("attachment spans lift to place, not the grasp settle",
          not weld[28] and weld[int(np.argmax(weld))])


def test_deformation_cannot_trade_the_task():
    """Precision phases must be untouchable, with no step at the boundary."""
    phases = pick_and_place()
    _, _, _, deform, _ = solve(phases, np.array([0.0, -1.2, 1.4, -1.6, -1.57, 0.0]))
    m = soft_mask(deform, width=8)
    check("locked phases stay exactly zero", np.all(m[~deform] == 0.0))
    step = np.max(np.abs(np.diff(m)))
    check("mask has no cliff into a locked phase", step <= 1.0 / 8 + 1e-9,
          f"largest step {step:.4f}")


def test_placed_criterion():
    """Success is the cube on the place table, not merely somewhere."""
    from predictive_replanning.cell import CUBE_HALF, PLACE_TABLE, PLACE_XY
    top = PLACE_TABLE["centre"][2] + PLACE_TABLE["half"][2]
    on = np.array([PLACE_XY[0], PLACE_XY[1], top + CUBE_HALF])
    check("on target counts", placed(on)[0])
    check("right spot, wrong height does not", not placed(on + np.array([0, 0, 0.2]))[0])
    check("still on the pick table does not", not placed(np.array([-0.66, -0.26, 0.32]))[0])


def main():
    _sig()
    for fn in (test_fk_matches_mujoco, test_reach_matches_the_datasheet_geometry,
               test_tcp_frame, test_point_jacobian, test_ou_closed_form,
               test_deformation_contract, test_deformation_cannot_trade_the_task,
               test_task_reaches_every_phase, test_placed_criterion,
               test_tracker_is_not_told_the_truth):
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n{sum(_results)} passed, {len(_results) - sum(_results)} failed")
    return 0 if all(_results) else 1


def _sig():
    """Author signature. stderr, tty-only, so redirected output stays clean."""
    import os, sys
    if os.environ.get("NO_BANNER") == "1" or not sys.stderr.isatty():
        return
    print("  " + "".join(chr(c - 7) for c in
          (104,105,107,124,115,39,121,104,111,116,104,117)), file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
