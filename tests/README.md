# Detection tests

The demo either sees the hand or it doesn't. This measures which, and where the
edges are.

```bash
python3 tests/test_detection.py      # ~10 s, writes tests/results.json
```

## What is real and what is not

**Real:** the detector. `harness.py` loads `reactive_replanning.py` and calls its
own `_cloud_cb`. Depth gate, TF batch transform, workspace box, the colour mask,
the geometric self-filter, the baseline median, the threshold and the debounce
counter are all the shipped code. Only two things are faked — `read_points_numpy`
and the TF buffer — because those are the ROS boundary, not the algorithm.

**Not real:** the sensor. `scene.py` synthesises the point cloud — a table, the
arm as capsules along the true `ARM_LINK_PAIRS` chain, and a foreign object at a
known position, with Gaussian range noise. There is no occlusion shadow, no
RealSense speckle, no lighting variation. So treat these as the detector's
behaviour *given clean input*: they bound the best case and locate the geometric
cliffs exactly. Real-world rates will be worse, particularly the colour filter,
which is the part most exposed to lighting.

Ground truth is known for every frame, so misses are counted, not estimated.

## Results

Seed `20260807`. Obstacle is skin-coloured; points scale with cross-section.

### Detection rate vs obstacle size

| Radius | Detected | Missed | Latency | Centroid error |
| ---: | ---: | ---: | ---: | ---: |
| 2 cm | 0% | 40/40 | — | — |
| 3 cm | 0% | 40/40 | — | — |
| 4 cm | **100%** | 0/40 | 0.10 s | 2.3 mm |
| 6 cm | **100%** | 0/40 | 0.10 s | 2.1 mm |
| 8 cm | **100%** | 0/40 | 0.10 s | 2.1 mm |
| 10 cm | **100%** | 0/40 | 0.10 s | 2.1 mm |

Nothing smaller than about 4 cm registers, because a small object returns fewer
points than `OBSTACLE_THRESHOLD`. Above that the detector is saturated — size
stops mattering and the centroid lands within ~2 mm of truth. Latency is two
frames at 20 Hz, which is `DEBOUNCE_FRAMES` exactly.

### Detection rate vs visible returns

| Foreign points in workspace | Detected |
| ---: | ---: |
| 60 | 0% |
| 100 | 0% |
| 130 | **100%** |
| 160 | **100%** |
| 400 | **100%** |

The transition sits between 100 and 130, which is `OBSTACLE_THRESHOLD = 120`
behaving as written. The step is sharp rather than gradual — there is no soft
region where detection is unreliable, which is the useful property.

### The blind spot

An obstacle walked in toward `forearm_link`:

| Distance to nearest link | Detected |
| ---: | ---: |
| 8 cm | **0%** |
| 10 cm | **0%** |
| 11 cm | 76% |
| 12 cm | 100% |
| 16 cm | 100% |
| 30 cm | 100% |

This is the number worth knowing. `ARM_SEG_RADIUS = 0.12` carves a 12 cm tube
around the kinematic chain and everything inside it is discarded as self, so a
hand closer than about 11 cm to the arm is invisible — not "detected late",
invisible. The filter cannot tell an operator's knuckles from the robot's own
housing at that range.

It is a deliberate trade: shrinking the tube brings back phantom detections off
the arm itself. But it means the safety story is "keeps clear of things it can
see", and things within 11 cm are not in that set.

### False positives

0 spurious detections in 300 frames with the arm in view and no obstacle
present. The colour mask plus the swept-volume history is doing its job on clean
input.

### Colour filter, in isolation

| | |
| --- | ---: |
| Skin points kept (recall on a hand) | 92.3% |
| Robot points rejected (gray / black / teal) | 85.5% |

The 14.5% of robot-coloured points that survive are what the geometric
self-filter exists to catch. Neither stage is sufficient alone.

### Throughput

3,770 points per frame, median **2.47 ms**, p95 2.73 ms. The 20 Hz throttle
allows 50 ms, so the pipeline runs with about **20× headroom** — detection
latency is set by the debounce counter, not by compute.

## Files

| | |
| --- | --- |
| `scene.py` | Synthetic camera frames with ground truth |
| `harness.py` | ROS stubs; loads the node and drives `_cloud_cb` |
| `test_detection.py` | The five measurements above |
| `results.json` | Machine-readable output |

## Not covered

Replan path length and planning time need a live MoveIt, so they are not
measured here. Nothing in this directory touches `plan_cartesian`,
`plan_arc_detour` or `compute_ik_solutions`.
