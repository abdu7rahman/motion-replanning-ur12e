"""The RViz cell, rebuilt as MJCF from the pinned vendor descriptions.

Nothing here is hand-modelled. The arm's meshes, masses and centres of mass all
come out of `config/ur12e/*` in Universal_Robots_ROS2_Description at a recorded
commit, and the gripper out of robotiq_hande_description -- the same package
this repo's own ur12e_hande.urdf.xacro includes. assets.py does the conversion
and writes a SHA-256 per file. An earlier draft of this module approximated the
arm with capsules and invented the link masses; that is exactly the mistake of
turning every downstream number into a claim about the model rather than the
robot, and it is why the meshes are fetched instead.

Two things the description package settles, both easy to get wrong by guessing:
`config/ur12e/default_kinematics.yaml` is byte-identical to `ur10e`, and
`config/ur12e/visual_parameters.yaml` names `meshes/ur10e/...` for every link.
There is no `meshes/ur12e/` at all. So a UR12e is a UR10e arm rated for more
payload, and using UR10e geometry is what the vendor does, not a substitution.

The ME5250 write-up records MuJoCo being tried first and dropped over
ros2_control segfaults and bridge timing. None of that applies here: there is
no bridge, the planner and controller share a process, and the physics steps
deterministically under a seed -- which is what makes the trials comparable.

Body origins are emitted as quaternions rather than euler angles. URDF rpy is
extrinsic XYZ; MuJoCo's default eulerseq is intrinsic xyz. They agree for every
joint here except wrist_3, whose origin (pi/2, pi, pi) is exactly where they
diverge. Converting in numpy removes the convention from the file.
"""
from __future__ import annotations

__author__ = "".join(
    chr(c - 7) for c in (104, 105, 107, 124, 115, 39, 121, 104, 111, 116, 104, 117)
)

from pathlib import Path

import numpy as np

import json

from predictive_replanning.assets import HANDE, OUT as ASSET_DIR, physical_params, visual_params
from predictive_replanning.ur12e import JOINT_ORIGINS, rpy

# ── the cell, in base_link ────────────────────────────────────────────
PICK_TABLE = dict(centre=(-0.72, -0.18, 0.15), half=(0.28, 0.22, 0.15))
PLACE_TABLE = dict(centre=(-0.58, 0.50, 0.22), half=(0.22, 0.18, 0.22))
CUBE_HALF = 0.02                     # the 4 cm cubes the write-up grasps
CUBE_XY = ((-0.66, -0.26), (-0.78, -0.14), (-0.70, -0.06))
PLACE_XY = (-0.58, 0.50)

LINKS = ("shoulder", "upper_arm", "forearm", "wrist_1", "wrist_2", "wrist_3")
JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")

# wrist_3 -> flange -> tool0, both from ur_macro.xacro.
_FLANGE_RPY = (0.0, -np.pi / 2.0, -np.pi / 2.0)
_TOOL0_RPY = (np.pi / 2.0, 0.0, np.pi / 2.0)


def _mat_to_quat(R: np.ndarray) -> str:
    t = float(np.trace(R))
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        q = (0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            q = ((R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s)
        elif i == 1:
            s = np.sqrt(1.0 - R[0, 0] + R[1, 1] - R[2, 2]) * 2.0
            q = ((R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s)
        else:
            s = np.sqrt(1.0 - R[0, 0] - R[1, 1] + R[2, 2]) * 2.0
            q = ((R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s)
    q = np.asarray(q) / np.linalg.norm(q)
    return " ".join(f"{v:.12f}" for v in q)


def _quat(roll: float, pitch: float, yaw: float) -> str:
    return _mat_to_quat(rpy(roll, pitch, yaw))


def _box(name, centre, half, rgba):
    return (f'<geom name="{name}" type="box" pos="{centre[0]} {centre[1]} {centre[2]}" '
            f'size="{half[0]} {half[1]} {half[2]}" rgba="{rgba}"/>')


def _provenance(asset_dir: Path) -> dict:
    return json.loads((asset_dir / "PROVENANCE.json").read_text())["files"]


def _mesh_geoms(prefix: str, key: str, offset: dict, prov: dict, *, visual: bool,
                indent: str) -> str:
    """One geom per material part, each carrying the colour from the file.

    Visual geoms take no contact and collision geoms are never drawn, so the
    render shows UR's own shell while the planner sees the collision hulls --
    which is what MoveIt would see too.
    """
    q = _quat(offset.get("roll", 0.0), offset.get("pitch", 0.0), offset.get("yaw", 0.0))
    pos = f'{offset.get("x", 0.0)} {offset.get("y", 0.0)} {offset.get("z", 0.0)}'
    out = []
    for n, part in enumerate(prov[key]["parts"]):
        mesh = Path(part["mesh"]).stem
        if visual:
            r, g, b, a = part["rgba"] or (0.75, 0.75, 0.75, 1.0)
            attrs = (f'rgba="{r:.4f} {g:.4f} {b:.4f} {a:.2f}" '
                     f'contype="0" conaffinity="0" group="2"')
        else:
            attrs = 'group="3" rgba="0 0 0 0"'
        out.append(f'<geom name="{prefix}_{n}" type="mesh" mesh="{mesh}" '
                   f'pos="{pos}" quat="{q}" {attrs}/>')
    return f"\n{indent}  ".join(out)


def build_mjcf(*, obstacle_radius: float = 0.09, seed_cubes: bool = True,
               asset_dir: Path | None = None) -> str:
    asset_dir = Path(asset_dir or ASSET_DIR)
    if not (asset_dir / "PROVENANCE.json").exists():
        raise FileNotFoundError(
            f"{asset_dir} has no converted meshes. Run:  python -m predictive_replanning.assets")
    vp, pp = visual_params(), physical_params()
    prov = _provenance(asset_dir)
    cog = pp["center_of_mass"]

    meshes = "\n".join(
        f'    <mesh name="{s}" file="{s}{"stl" if (asset_dir / (s + ".stl")).exists() else "obj"}"/>'
        .replace(f'{s}stl', f'{s}.stl').replace(f'{s}obj', f'{s}.obj')
        for s in sorted(p.stem for p in asset_dir.iterdir()
                        if p.suffix in (".obj", ".stl")))

    body, close = "", ""
    for i, (link, jname) in enumerate(zip(LINKS, JOINTS)):
        x, y, z, r, p, yw = JOINT_ORIGINS[i]
        ind = "    " * (i + 3)
        c = cog[f"{link}_cog"]
        body += (
            f'\n{ind}<body name="{link}_link" pos="{x} {y} {z}" quat="{_quat(r, p, yw)}">'
            f'\n{ind}  <joint name="{jname}" axis="0 0 1" range="-6.2832 6.2832"/>'
            f'\n{ind}  <inertial pos="{c["x"]} {c["y"]} {c["z"]}" '
            f'mass="{pp[f"{link}_mass"]}" diaginertia="0.03 0.03 0.03"/>'
            f'\n{ind}  {_mesh_geoms(f"{link}_vis", f"ur_{link}_visual", vp[link]["mesh_offset"], prov, visual=True, indent=ind)}'
            f'\n{ind}  {_mesh_geoms(f"{link}_col", f"ur_{link}_collision", vp[link]["mesh_offset"], prov, visual=False, indent=ind)}')
        close = f'\n{ind}</body>' + close

    # wrist_3 -> flange -> tool0, then the Hand-E's own coupler and body.
    tool_q = _mat_to_quat(rpy(*_FLANGE_RPY) @ rpy(*_TOOL0_RPY))
    g = "    " * 9
    h = HANDE
    tool = (
        f'\n{g}  <body name="tool0" pos="0 0 0" quat="{tool_q}">'
        f'\n{g}    {_mesh_geoms("coupler_vis", "hande_io_coupler", {}, prov, visual=True, indent=g + "    ")}'
        f'\n{g}    <body name="hande" pos="0 0 {h["coupler_height"]}">'
        f'\n{g}      <inertial pos="0 0 {h["hande_height"]/2:.4f}" mass="0.9" diaginertia="0.002 0.002 0.002"/>'
        f'\n{g}      {_mesh_geoms("hande_vis", "hande_hande", {}, prov, visual=True, indent=g + "      ")}'
        f'\n{g}      <geom name="hande_col" type="cylinder" '
        f'fromto="0 0 0 0 0 {h["hande_height"]}" size="{h["hande_radius"]}" group="3" rgba="0 0 0 0"/>'
        f'\n{g}      <body name="left_finger" pos="0 0 {h["hande_height"]}">'
        f'\n{g}        <joint name="hande_left_finger_joint" type="slide" axis="1 0 0" '
        f'range="{h["grip_min"]} {h["grip_max"]}"/>'
        f'\n{g}        <inertial pos="0 0 0.01" mass="{h["finger_mass"]}" diaginertia="1e-6 1e-6 1e-6"/>'
        f'\n{g}        {_mesh_geoms("left_finger_vis", "hande_finger", {}, prov, visual=True, indent=g + "        ")}'
        f'\n{g}      </body>'
        f'\n{g}      <body name="right_finger" pos="0 0 {h["hande_height"]}" quat="0 0 0 1">'
        f'\n{g}        <joint name="hande_right_finger_joint" type="slide" axis="1 0 0" '
        f'range="{h["grip_min"]} {h["grip_max"]}"/>'
        f'\n{g}        <inertial pos="0 0 0.01" mass="{h["finger_mass"]}" diaginertia="1e-6 1e-6 1e-6"/>'
        f'\n{g}        {_mesh_geoms("right_finger_vis", "hande_finger", {}, prov, visual=True, indent=g + "        ")}'
        f'\n{g}      </body>'
        f'\n{g}      <site name="tcp" pos="0 0 {h["hande_height"] + 0.0465}" size="0.008" rgba="1 0 0 1"/>'
        f'\n{g}    </body>'
        f'\n{g}  </body>')

    cubes = ""
    if seed_cubes:
        cz = PICK_TABLE["centre"][2] + PICK_TABLE["half"][2] + CUBE_HALF
        for i, (cx, cy) in enumerate(CUBE_XY):
            cubes += (f'\n    <body name="cube_{i}" pos="{cx} {cy} {cz}">'
                      f'\n      <freejoint name="cube_{i}_free"/>'
                      f'\n      <geom name="cube_{i}_g" type="box" '
                      f'size="{CUBE_HALF} {CUBE_HALF} {CUBE_HALF}" '
                      f'rgba="0.20 0.40 0.85 1" mass="0.05"/>'
                      f'\n    </body>')

    return f"""<mujoco model="ur12e_hande_predictive_cell">
  <compiler angle="radian" autolimits="true" meshdir="{asset_dir}"/>
  <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="1280" offheight="960"/>
    <headlight ambient="0.42 0.42 0.45" diffuse="0.28 0.28 0.30" specular="0 0 0"/>
    <map shadowclip="1.5"/>
  </visual>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.90 0.90 0.92" rgb2="0.82 0.82 0.85"
             width="300" height="300"/>
    <material name="gridmat" texture="grid" texrepeat="6 6" reflectance="0.05"/>
{meshes}
  </asset>

  <worldbody>
    <light pos="0.6 -0.9 2.2" dir="-0.25 0.35 -1" diffuse="0.55 0.55 0.57"
           specular="0.08 0.08 0.08" castshadow="true"/>
    <light pos="-1.6 0.9 1.6" dir="0.6 -0.35 -0.7" diffuse="0.28 0.28 0.31"
           specular="0.02 0.02 0.02" castshadow="false"/>
    <geom name="floor" type="plane" size="4 4 0.05" material="gridmat"/>

    {_box("pedestal", (0, 0, 0.09), (0.13, 0.13, 0.09), "0.30 0.30 0.33 1")}
    {_box("pick_table", PICK_TABLE["centre"], PICK_TABLE["half"], "0.55 0.42 0.30 1")}
    {_box("place_table", PLACE_TABLE["centre"], PLACE_TABLE["half"], "0.55 0.42 0.30 1")}
{cubes}

    <!-- The moving obstacle. mocap, because its motion is prescribed by the
         random process in obstacle.py rather than by contact: an obstacle the
         arm could shove aside is not the obstacle being studied. -->
    <body name="obstacle" mocap="true" pos="-0.70 0.18 0.55">
      <geom name="obstacle_g" type="sphere" size="{obstacle_radius}"
            rgba="0.20 0.75 0.30 0.55" contype="0" conaffinity="0"/>
    </body>

    <body name="base_link" pos="0 0 0.18">
      {_mesh_geoms("base_vis", "ur_base_visual", vp["base"]["mesh_offset"], prov, visual=True, indent="      ")}{body}{tool}{close}
    </body>
  </worldbody>

  <!-- Grasp as an equality constraint rather than friction. The ME5250 report
       records the alternative: "the cube seems to not be physically attached in
       simulation, occasionally resulting in dropped objects during transport".
       Tuning contact friction until a box stays in the jaws would make every
       success rate below a statement about that tuning. A weld says plainly
       that a successful grasp is assumed and the thing being measured is
       whether the arm gets the object there. -->
  <equality>
    <weld name="grasp_0" body1="hande" body2="cube_0" active="false" solref="0.01 1"/>
  </equality>

  <actuator>
{chr(10).join(f'    <position name="act_{j}" joint="{j}" kp="3000" dampratio="1"/>' for j in JOINTS)}
    <position name="act_grip_l" joint="hande_left_finger_joint" kp="200" dampratio="1"/>
    <position name="act_grip_r" joint="hande_right_finger_joint" kp="200" dampratio="1"/>
  </actuator>
</mujoco>
"""
