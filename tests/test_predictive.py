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
from predictive_replanning.obstacle import (ObstacleProcess,        # noqa: E402
                                            record_intercept_track)
from predictive_replanning.predict import arm_points, ObstacleTracker  # noqa: E402
from predictive_replanning.replan import (_limit_payload, _window,   # noqa: E402
                                          deform_minimal, deform_optimise,
                                          nominal_trajectory, soft_mask)
from predictive_replanning.task import (grip_grasp, grip_open,       # noqa: E402
                                        pick_and_place, placed, solve)

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
    # A test wants a fixed motion, so it passes a seeded generator. That is
    # the caller's choice now rather than something the process imposes.
    proc = ObstacleProcess(rng=np.random.default_rng(1))
    for _ in range(300):
        proc.step(0.01)
    proc.box_half = np.array([9.0, 9.0, 9.0])          # walls off; closed form has none
    H = np.array([0.25, 1.0])
    mean, std = proc.true_forecast(H)
    samples = {h: [] for h in H}
    for k in range(1500):
        q = ObstacleProcess(rng=np.random.default_rng(90_000 + k))
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
    proc = ObstacleProcess(rng=np.random.default_rng(4))
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
    check("jaws are closed while carrying", float(np.max(grip[weld])) < float(np.max(grip)))
    check("carrying starts at the lift, after the close settles", not weld[28])


def test_deformation_cannot_trade_the_task():
    """Precision phases must be untouchable, with no step at the boundary."""
    phases = pick_and_place()
    _, _, _, deform, _ = solve(phases, np.array([0.0, -1.2, 1.4, -1.6, -1.57, 0.0]))
    m = soft_mask(deform, width=8)
    check("locked phases stay exactly zero", np.all(m[~deform] == 0.0))
    step = np.max(np.abs(np.diff(m)))
    check("mask has no cliff into a locked phase", step <= 1.0 / 8 + 1e-9,
          f"largest step {step:.4f}")


def test_gripper_is_derived_not_typed():
    """Jaw commands must come from the object, and the sense must be right.

    Travel 0 is CLOSED and 0.025 is a 50 mm opening -- the gap is exactly twice
    the travel, measured off the vendor collision meshes. An earlier version had
    open and closed the wrong way round, which together with fingers that
    carried no collision geometry is why nothing could be picked up.
    """
    from predictive_replanning.cell import CUBE_HALF
    w = 2.0 * CUBE_HALF
    check("open clears the object", 2 * grip_open(w) > w,
          f"gap {2 * grip_open(w):.4f} m vs object {w:.4f} m")
    check("grasp squeezes the object", 2 * grip_grasp(w) < w,
          f"gap {2 * grip_grasp(w):.4f} m vs object {w:.4f} m")
    check("open is wider than grasp", grip_open(w) > grip_grasp(w))
    check("both inside the Hand-E stroke",
          0.0 <= grip_grasp(w) and grip_open(w) <= 0.025)
    wide = grip_open(0.06)
    check("a bigger object opens the jaws wider or saturates", wide >= grip_open(w))


def test_fingers_can_actually_collide():
    """Every finger needs collision geometry, and there must be no weld."""
    model = mujoco.MjModel.from_xml_string(build_mjcf())
    fingers = [g for g in range(model.ngeom)
               if "finger_col" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
    check("fingers have collision geoms", len(fingers) >= 2, f"{len(fingers)} geoms")
    check("and they are collidable", all(model.geom_contype[g] for g in fingers))
    check("the grasp is contact, not a weld", model.neq == 0,
          f"neq={model.neq}")


def test_tool_stays_vertical():
    """The tool must be straight down everywhere, not only at phase ends.

    Interpolating joint angles between two solved poses pinned the tool at the
    endpoints and let it swing 9.8 degrees off vertical in between, so the
    "top-down" grasp was top-down twice per phase. The trajectory is solved in
    task space for this reason.
    """
    traj, _, _, _, _ = solve(pick_and_place(), np.array([0.0, -1.2, 1.4, -1.6, -1.57, 0.0]))
    tilts = [np.degrees(np.arccos(np.clip(-ur12e.fk_tcp(q)[:3, 2][2], -1, 1))) for q in traj]
    check("tool vertical at every waypoint", max(tilts) < 1.0,
          f"max {max(tilts):.4f} deg, mean {np.mean(tilts):.4f} deg")


def test_deformation_does_not_tip_the_tool():
    """A dodge must not tilt the gripper; that is how a friction grasp drops."""
    traj, times, _, deform, holding = solve(
        pick_and_place(), np.array([0.0, -1.2, 1.4, -1.6, -1.57, 0.0]))
    allow = soft_mask(deform)
    mid = int(np.median(np.where(deform)[0]))
    on = arm_points(traj[mid])[0][6]

    def tilt(tr):
        return max(np.degrees(np.arccos(np.clip(-ur12e.fk_tcp(q)[:3, 2][2], -1, 1))) for q in tr)

    for name, fn in (("minimal", deform_minimal), ("optimise", deform_optimise)):
        trk = ObstacleTracker(on)
        for _ in range(30):
            trk.update(on, 0.02)
        new, dev, _, _ = fn(traj, times, 0.0, trk, base_radius=0.09, n_sigma=1.0,
                            clearance=0.02, horizon=1.0, sigma_cap=0.1,
                            allow=allow, carrying=holding)
        check(f"{name}: tool still vertical after deforming", tilt(new) < 1.0,
              f"max {tilt(new):.3f} deg")
        check(f"{name}: it actually moved something", dev > 0.0, f"{dev:.3f} rad")


def test_payload_bound_is_relative():
    """The bound limits the speed-up over the plan, not absolute speed.

    An absolute cap throttled every deformation: the nominal carry already
    moves the tool 0.11 m between waypoints, so a few-centimetre ceiling scaled
    every correction to a third of itself and nothing ever cleared.
    """
    traj, _, _, _, holding = solve(pick_and_place(),
                                   np.array([0.0, -1.2, 1.4, -1.6, -1.57, 0.0]))
    zero = np.zeros_like(traj)
    check("a zero deformation is never scaled",
          np.allclose(_limit_payload(traj, traj, zero, holding), 0.0))
    # A *uniform* offset shifts every waypoint alike, so the tool's speed
    # between waypoints is unchanged and the bound correctly leaves it alone --
    # the bound is on speed-up, not on displacement. To trip it the
    # perturbation has to vary along the trajectory, so this one alternates.
    flat = np.zeros_like(traj)
    flat[holding] += 0.35
    check("a uniform offset is not a speed-up, so it passes",
          np.allclose(_limit_payload(traj, traj, flat, holding), flat))

    big = np.zeros_like(traj)
    sign = np.where(np.arange(len(traj)) % 2 == 0, 1.0, -1.0)
    big[holding] += 0.35 * sign[holding, None]
    out = _limit_payload(traj, traj, big, holding)
    check("a deformation that outruns the plan is scaled back",
          float(np.abs(out).max()) < float(np.abs(big).max()),
          f"{np.abs(big).max():.3f} -> {np.abs(out).max():.3f} rad")


def test_arm_radii_match_the_meshes():
    """The planner's link thicknesses come off the vendor meshes, not a guess.

    LINK_RADIUS is a hard-coded table so the planner does not need MuJoCo to
    run. This re-derives it from the compiled model: each collision mesh's
    half-extent across its shaft has to fit inside the radius the planner
    assumes for that segment, or the skeleton is thinner than the robot and
    every clearance it computes is optimistic.
    """
    from predictive_replanning.predict import LINK_RADIUS, arm_points, arm_radii
    from predictive_replanning.run import _COL_GEOM, _model_for, _arm_qpos_adr, BASE_Z
    model = _model_for(0.07)
    # Each link's mesh, measured across the shaft (the two axes that are not
    # the long one), against the radius the planner gives that segment.
    want = {"shoulder": 0, "upper_arm": 2, "forearm": 3,
            "wrist_1": 4, "wrist_2": 5, "wrist_3": 6}
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        body = name.rsplit("_col", 1)[0]
        if not _COL_GEOM.match(name) or body not in want or model.geom_dataid[g] < 0:
            continue
        mid = model.geom_dataid[g]
        V = model.mesh_vert[model.mesh_vertadr[mid]:
                            model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        half = np.abs(V.astype(float)).max(0)
        across = float(np.sort(half)[:2].max())        # drop the long axis
        r = LINK_RADIUS[want[body]]
        check(f"{body} radius covers its mesh", across <= r + 1e-3,
              f"mesh {across:.4f} m vs assumed {r:.4f} m")

    # And the whole cover, end to end: the skeleton-plus-radii distance must
    # never claim more room than MuJoCo measures between the real surfaces.
    data = mujoco.MjData(model)
    adr = _arm_qpos_adr(model)
    geoms = [g for g in range(model.ngeom)
             if _COL_GEOM.match(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
    og = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle_g")
    mocap = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obstacle")]
    rng = np.random.default_rng(3)
    lo = np.array([-3.14, -2.6, -2.6, -3.14, -3.14, -3.14])
    hi = np.array([3.14, 0.2, 2.6, 3.14, 3.14, 3.14])
    err = []
    for _ in range(400):
        q = rng.uniform(lo, hi)
        for a, v in zip(adr, q):
            data.qpos[a] = v
        pts = arm_points(q)[0]
        p = pts[rng.integers(len(pts))] + rng.normal(0.0, 0.18, 3)
        data.mocap_pos[mocap] = p + np.array([0.0, 0.0, BASE_Z])
        mujoco.mj_forward(model, data)
        true = min(float(mujoco.mj_geomDistance(model, data, og, g, 3.0, None))
                   for g in geoms)
        skel = float((np.linalg.norm(pts - p, axis=1) - arm_radii()).min()) - 0.07
        err.append(skel - true)
    # And that the radii really are pose-independent, which is what lets them be
    # computed once: consecutive joint origins are a fixed URDF translation, so
    # the axial spacing the radii widen for cannot move when the arm does.
    spans = []
    for _ in range(50):
        q = rng.uniform(lo, hi)
        pts = arm_points(q)[0]
        spans.append(np.linalg.norm(np.diff(pts[:8], axis=0), axis=1))
    spans = np.array(spans)
    spread = float((spans.max(0) - spans.min(0)).max())
    check("skeleton segment lengths do not depend on the pose", spread < 1e-9,
          f"worst spread {spread:.2e} m")

    err = np.array(err)
    check("the skeleton no longer flatters itself", err.mean() < 0.0,
          f"mean overestimate {err.mean():+.4f} m, was +0.065 m without radii")
    check("worst case is bounded", err.max() < 0.20, f"max {err.max():.4f} m")


def test_clearance_sees_the_whole_arm():
    """The reported metric has to cover the robot, not one link of it.

    UR ships one collision mesh per link and the loader splits meshes by
    material, so the geoms come out as `<link>_col_<n>` while the Hand-E shell
    is a bare `hande_col`. Selecting them with endswith("_col") matched exactly
    one geom out of 67: every clearance reported was the distance to the
    gripper, with the forearm, both wrists and the finger pads invisible. The
    arm could be driven through the obstacle and the number would not move.
    """
    from predictive_replanning.run import _COL_GEOM, _model_for
    model = _model_for(0.07)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
             for g in range(model.ngeom)]
    sel = [n for n in names if _COL_GEOM.match(n)]
    for link in ("shoulder", "upper_arm", "forearm", "wrist_1", "wrist_2",
                 "wrist_3", "hande", "left_finger", "right_finger"):
        check(f"{link} is in the clearance metric",
              any(n.startswith(link + "_col") for n in sel))
    check("no visual mesh sneaks in", not any("_vis" in n for n in sel))
    check("nothing off the robot", not any(n.startswith(("cube", "obstacle", "floor"))
                                           for n in sel), f"{len(sel)} geoms")


def test_obstacle_actually_intercepts():
    """The obstacle has to reach the path, or the comparison measures luck.

    Every earlier version of this got in the way only some of the time -- the
    box wanderer about two runs in three, the tail chase about the same -- and
    the runs where nothing happened dragged every strategy toward the same
    number. The property under test is that a do-nothing arm is hit, on every
    draw, without the obstacle being teleported onto it.

    Checked on a synthetic straight-line carry so it is geometry rather than a
    simulator run: 60 independent draws, each of which must close from its
    standoff to inside the sphere.
    """
    dt, n = 0.02, 400
    t = np.arange(n) * dt
    path = np.stack([-0.75 + 0.0 * t, -0.30 + 0.09 * t, 0.55 + 0.02 * np.sin(t)], 1)
    carry = (t >= 2.0) & (t <= 6.0)
    radius = 0.07
    rng = np.random.default_rng(4)
    gaps, starts = [], []
    for _ in range(60):
        tk = record_intercept_track(tcp_path=path, times=t, carry_mask=carry,
                                    dt=dt, steps=n, radius=radius, theta=1.4,
                                    sigma=0.28, meas_std=0.02, rng=rng)
        gaps.append(np.linalg.norm(tk.positions - path, axis=1).min())
        starts.append(np.linalg.norm(tk.positions[0] - path[0]))
    gaps, starts = np.array(gaps), np.array(starts)
    check("every draw reaches the tool", gaps.max() < radius,
          f"worst closest approach {gaps.max():.4f} m, radius {radius}")
    check("it starts well clear of the arm", starts.min() > 0.15,
          f"nearest start {starts.min():.3f} m")
    check("it is not parked on the path", np.median(gaps) > 1e-3,
          f"median closest approach {np.median(gaps):.4f} m")


def test_obstacle_is_recorded_not_reactive():
    """It is a recording, so an arm that deviates genuinely gets away.

    This is what keeps the experiment fair, and it is a structural property
    rather than a statistical one: the motion is drawn once, before any
    strategy runs, and every strategy replays the same array. A pursuer that
    re-aimed at the live arm would make every strategy fail by construction and
    the comparison would say nothing about prediction.

    Two things are checked. The draw is a pure function of the generator and
    the baseline path -- equal states give bit-identical motion. And a bounded
    sideways move clears it: displacing the path by 25 cm along the direction
    the replanner pushes takes the arm outside the sphere on every draw, so
    replanning is something the obstacle can actually be escaped by.
    """
    dt, n = 0.02, 400
    t = np.arange(n) * dt
    path = np.stack([-0.75 + 0.0 * t, -0.30 + 0.09 * t, 0.55 + 0.02 * np.sin(t)], 1)
    carry = (t >= 2.0) & (t <= 6.0)
    kw = dict(tcp_path=path, times=t, carry_mask=carry, dt=dt, steps=n,
              radius=0.07, theta=1.4, sigma=0.28, meas_std=0.02)
    a = record_intercept_track(rng=np.random.default_rng(7), **kw)
    b = record_intercept_track(rng=np.random.default_rng(7), **kw)
    check("the same draw is the same motion",
          np.array_equal(a.positions, b.positions))

    rng = np.random.default_rng(12)
    worst = -np.inf
    for _ in range(40):
        tk = record_intercept_track(rng=rng, **kw)
        d = np.linalg.norm(tk.positions - path, axis=1)
        s = int(d.argmin())
        away = path[s] - tk.positions[s]
        nrm = np.linalg.norm(away)
        away = away / nrm if nrm > 1e-9 else np.array([0.0, 0.0, 1.0])
        dodged = path + 0.25 * away
        gap = np.linalg.norm(tk.positions - dodged, axis=1).min()
        worst = max(worst, 0.07 - gap)
    check("a 25 cm sidestep escapes it", worst < 0.0,
          f"deepest overlap with the sidestepped path {max(worst, 0.0):.4f} m")


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
               test_task_reaches_every_phase, test_gripper_is_derived_not_typed,
               test_fingers_can_actually_collide, test_tool_stays_vertical,
               test_deformation_does_not_tip_the_tool, test_payload_bound_is_relative,
               test_placed_criterion, test_clearance_sees_the_whole_arm,
               test_arm_radii_match_the_meshes,
               test_obstacle_actually_intercepts,
               test_obstacle_is_recorded_not_reactive,
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
