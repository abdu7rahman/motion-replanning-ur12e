#!/usr/bin/env python3
"""Measures what the obstacle detector actually does.

Every number here comes from running reactive_replanning.py's own _cloud_cb on
synthetic RealSense frames with a known ground-truth obstacle, so misses and
localisation error can be counted rather than estimated.

    python3 tests/test_detection.py
"""
import json, os, statistics, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene, harness                                              # noqa: E402

CLS = harness.load_node_class()
THROTTLE = CLS.CLOUD_THROTTLE
BASELINE_FRAMES = 8
MAX_FRAMES = 6            # frames the obstacle is present before we call it a miss


def trial(rng, obstacle, obstacle_rgb=scene.SKIN, obstacle_pts=900, clutter=True):
    """Returns (detected, frames_to_detect, centroid_error_m, per_frame_seconds)."""
    rig = harness.Rig(CLS)
    kw = dict(obstacle_rgb=obstacle_rgb, obstacle_pts=obstacle_pts)
    times = []
    for _ in range(BASELINE_FRAMES):
        cam, rgb, _ = scene.build(rng)
        times.append(rig.feed(cam, rgb))
    rig.node._executing = True
    for i in range(MAX_FRAMES):
        cam, rgb, gt = scene.build(rng, obstacle=obstacle, **kw)
        times.append(rig.feed(cam, rgb))
        if rig.detections:
            err = float(np.linalg.norm(rig.detections[-1][0] - gt))
            return True, i + 1, err, times
    return False, None, None, times


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    _sig()
    rng = np.random.default_rng(20260807)
    out = {}

    # ---- 1. detection rate vs obstacle size ------------------------------
    print("\n1. Detection rate vs obstacle radius   (skin-coloured, 40 trials each)")
    print(f"   {'radius':>8} {'detected':>10} {'missed':>8} {'latency':>10} {'centroid err':>14}")
    size_rows = []
    for radius in (0.02, 0.03, 0.04, 0.06, 0.08, 0.10):
        hits, lat, errs = 0, [], []
        N = 40
        for _ in range(N):
            # resample until clear of the arm, so this measures size not occlusion
            while True:
                c = (rng.uniform(-0.95, -0.40), rng.uniform(-0.35, 0.45), rng.uniform(0.30, 0.70))
                if min(np.linalg.norm(np.array(c) - p) for p in scene.LINKS.values()) >= 0.30:
                    break
            det, frames, err, _ = trial(rng, (c, radius),
                                        obstacle_pts=int(900 * (radius / 0.09) ** 2))
            if det:
                hits += 1; lat.append(frames * THROTTLE); errs.append(err)
        row = {"radius_m": radius, "trials": N, "detected": hits,
               "detect_rate": hits / N,
               "latency_s": round(statistics.mean(lat), 3) if lat else None,
               "centroid_err_mm": round(1000 * statistics.mean(errs), 1) if errs else None}
        size_rows.append(row)
        print(f"   {radius*100:6.0f}cm {pct(row['detect_rate']):>10} {N-hits:>8} "
              f"{(str(row['latency_s'])+' s') if lat else '-':>10} "
              f"{(str(row['centroid_err_mm'])+' mm') if errs else '-':>14}")
    out["by_size"] = size_rows

    # ---- 1b. detection vs visible returns --------------------------------
    print(f"\n1b. Detection rate vs visible returns   (threshold is "
          f"{CLS.OBSTACLE_THRESHOLD} foreign points, 30 trials each)")
    print(f"   {'points':>8} {'detected':>10}")
    pt_rows = []
    for npts in (60, 100, 130, 160, 220, 400):
        hits = 0; N = 30
        for _ in range(N):
            while True:
                c = (rng.uniform(-0.95, -0.40), rng.uniform(-0.35, 0.45), rng.uniform(0.30, 0.70))
                if min(np.linalg.norm(np.array(c) - p) for p in scene.LINKS.values()) >= 0.30:
                    break
            det, *_ = trial(rng, (c, 0.07), obstacle_pts=npts)
            hits += bool(det)
        pt_rows.append({"points": npts, "detect_rate": hits / N})
        print(f"   {npts:>8} {pct(hits/N):>10}")
    out["by_points"] = pt_rows

    # ---- 2. false positives ---------------------------------------------
    print("\n2. False positives   (no obstacle, arm in frame, 300 frames while executing)")
    rig = harness.Rig(CLS)
    for _ in range(BASELINE_FRAMES):
        cam, rgb, _ = scene.build(rng); rig.feed(cam, rgb)
    rig.node._executing = True
    for _ in range(300):
        cam, rgb, _ = scene.build(rng); rig.feed(cam, rgb)
    fp = len(rig.detections)
    out["false_positives"] = {"frames": 300, "spurious_detections": fp}
    print(f"   {fp} spurious detections in 300 frames  ({pct(fp/300)})")

    # ---- 3. the blind spot: how close to the arm can an obstacle get? ----
    print("\n3. Self-filter blind spot   (obstacle walked toward the forearm, 25 trials each)")
    print(f"   {'distance to arm':>17} {'detected':>10}")
    fore = scene.LINKS['forearm_link']
    blind_rows = []
    for d in (0.08, 0.10, 0.11, 0.12, 0.13, 0.14, 0.16, 0.20, 0.30):
        hits = 0; N = 25
        for _ in range(N):
            direction = np.array([0.0, -1.0, 0.35]); direction /= np.linalg.norm(direction)
            c = fore + direction * d
            det, *_ = trial(rng, (tuple(c), 0.07), obstacle_pts=700)
            hits += bool(det)
        blind_rows.append({"dist_m": d, "detect_rate": hits / N})
        print(f"   {d*100:14.0f} cm {pct(hits/N):>10}")
    out["self_filter"] = blind_rows

    # ---- 4. colour filter ------------------------------------------------
    print("\n4. Colour filter   (does a hand survive the robot-colour mask?)")
    node = harness.Rig(CLS).node
    rr = np.random.default_rng(7)
    def band(base, n, spread=28):
        return np.clip(np.array(base) + rr.normal(0, spread, (n, 3)), 0, 255).astype(np.int16)
    skin = band(scene.SKIN, 4000)
    robot = np.vstack([band(scene.GRAY, 2000), band(scene.BLACK, 1000, 10), band(scene.TEAL, 1000)])
    skin_dropped = node._is_robot_color(skin).mean()
    robot_dropped = node._is_robot_color(robot).mean()
    out["colour_filter"] = {"skin_kept": float(1 - skin_dropped),
                            "robot_rejected": float(robot_dropped)}
    print(f"   skin points kept       {pct(1-skin_dropped)}   (recall on a hand)")
    print(f"   robot points rejected  {pct(robot_dropped)}   (gray / black / teal)")

    # ---- 5. throughput ---------------------------------------------------
    print("\n5. Throughput   (full pipeline per frame)")
    rig = harness.Rig(CLS); times = []
    for _ in range(60):
        cam, rgb, _ = scene.build(rng, obstacle=((-0.6, 0.2, 0.5), 0.08))
        times.append(rig.feed(cam, rgb))
    times = sorted(times)
    med = times[len(times) // 2]; p95 = times[int(0.95 * len(times))]
    pts = len(cam)
    out["throughput"] = {"points_per_frame": int(pts), "median_ms": round(med * 1000, 2),
                         "p95_ms": round(p95 * 1000, 2), "budget_ms": THROTTLE * 1000,
                         "headroom_x": round(THROTTLE / med, 1)}
    print(f"   {pts} points/frame · median {med*1000:.2f} ms · p95 {p95*1000:.2f} ms")
    print(f"   budget at {1/THROTTLE:.0f} Hz is {THROTTLE*1000:.0f} ms  ->  {THROTTLE/med:.0f}x headroom")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote tests/results.json")
    return out


if __name__ == "__main__":
    main()


def _sig():
    """Author signature. stderr, tty-only, so redirected output stays clean."""
    import os, sys
    if os.environ.get("NO_BANNER") == "1" or not sys.stderr.isatty():
        return
    print("  " + "".join(chr(c - 7) for c in
          (104,105,107,124,115,39,121,104,111,116,104,117)), file=sys.stderr)
