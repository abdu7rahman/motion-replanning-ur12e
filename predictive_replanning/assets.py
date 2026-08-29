"""Convert the pinned UR12e and Hand-E descriptions into MuJoCo assets.

Nothing here is modelled by hand. Both descriptions are cloned at a recorded
commit, every file that gets used is hashed, and the conversion is a format
change rather than a redraw. That is the same rule the bimanual workcell
states for itself, and it is worth restating: a hand-made part turns every
number downstream into a claim about the model instead of about the robot.

Two findings the description package settles, both of which look like guesses
until you go and read it:

  * `config/ur12e/default_kinematics.yaml` is byte-identical to `ur10e`.
  * `config/ur12e/visual_parameters.yaml` points every mesh path at
    `meshes/ur10e/...`. There is no `meshes/ur12e/` directory at all.

So the UR12e is a UR10e arm with a higher payload rating, and using UR10e
geometry for it is what the vendor's own description does, not a substitution
made here. Masses come from `config/ur12e/physical_parameters.yaml`.

MuJoCo reads STL and OBJ, not COLLADA, so the visual `.dae` files are
converted here. They are split by material rather than merged: one UR mesh
carries several, and a forearm alone binds four -- LinkGrey 0.82, JointGrey
0.278, Black 0.033 and URBlue (0.49, 0.678, 0.8). Merging them and painting
the link one flat colour, which an earlier draft of this module did, throws
away the arm's actual appearance and replaces it with a guess; the blue it
guessed was not even UR's blue. So each material becomes its own OBJ and its
own geom, and the RGBA comes from the file.

The collision `.stl` files are copied untouched.
"""
from __future__ import annotations

__author__ = "".join(
    chr(c - 7) for c in (104, 105, 107, 124, 115, 39, 121, 104, 111, 116, 104, 117)
)

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY = ROOT / "third_party"
OUT = ROOT / "assets"

SOURCES = {
    "ur_description": "https://github.com/UniversalRobots/Universal_Robots_ROS2_Description.git",
    "hande_description": "https://github.com/AGH-CEAI/robotiq_hande_description.git",
}

#: Hand-E xacro argument defaults, read from the file rather than assumed.
HANDE = dict(coupler_height=0.011, coupler_shell_height=0.0169,
             coupler_parent_cutoff=0.003, coupler_hande_cutoff=0.0029,
             hande_height=0.099, hande_radius=0.0375,
             finger_mass=0.03804, grip_min=0.0, grip_max=0.025,
             # Robotiq's rated grip force for the Hand-E, and the squeeze the
             # task commands. The actuator gain is derived from the pair rather
             # than picked: kp = force / squeeze. Left at a round kp the model
             # produced 2.2 N, an order of magnitude under the rated minimum,
             # which held the cube only while the arm barely moved and threw it
             # across the cell once the arm actually articulated.
             grip_force_min_n=20.0, grip_force_max_n=185.0,
             grip_force_nominal_n=20.0, grip_squeeze_m=0.0025)


class _DegreesLoader(yaml.SafeLoader):
    """visual_parameters.yaml tags angles with `!degrees`."""


_DegreesLoader.add_constructor(
    "!degrees", lambda loader, node: float(np.deg2rad(float(loader.construct_scalar(node)))))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def visual_params() -> dict:
    p = THIRD_PARTY / "ur_description" / "config" / "ur12e" / "visual_parameters.yaml"
    return yaml.load(p.read_text(), Loader=_DegreesLoader)["mesh_files"]


def physical_params() -> dict:
    p = THIRD_PARTY / "ur_description" / "config" / "ur12e" / "physical_parameters.yaml"
    return yaml.safe_load(p.read_text())["inertia_parameters"]


def split_by_material(dae: Path) -> dict:
    """COLLADA -> {material name: (vertices, faces, rgba)}, scene transforms applied.

    Grouped by the primitive's bound material instead of merged, because a
    single UR mesh binds several and the join would lose all of them.
    """
    import collada

    doc = collada.Collada(str(dae))
    groups: dict[str, list] = {}
    colours: dict[str, tuple] = {}
    for obj in doc.scene.objects("geometry"):
        for prim in obj.primitives():
            mat = prim.material
            name = getattr(mat, "id", None) or str(mat) or "default"
            name = name.replace("-material", "").replace("-", "_")
            diffuse = getattr(getattr(mat, "effect", None), "diffuse", None)
            if isinstance(diffuse, (tuple, list, np.ndarray)) and len(diffuse) >= 3:
                colours[name] = tuple(float(c) for c in diffuse[:3]) + (1.0,)
            else:
                # No bound diffuse. Mid grey, and recorded as such in the
                # provenance so it is visible rather than assumed.
                colours.setdefault(name, (0.75, 0.75, 0.75, 1.0))
            # Not every primitive is a triangle set -- UR's exports include
            # polylists of quads, which reshape(-1, 3) silently cannot take.
            tri = prim.triangleset() if hasattr(prim, "triangleset") else prim
            idx = np.asarray(tri.vertex_index)
            if idx.ndim == 1:
                idx = idx.reshape(-1, 3)
            groups.setdefault(name, []).append(
                (np.asarray(tri.vertex, dtype=float), idx))

    out = {}
    for name, chunks in groups.items():
        verts, faces, base = [], [], 0
        for v, f in chunks:
            verts.append(v)
            faces.append(f + base)
            base += len(v)
        out[name] = (np.vstack(verts), np.vstack(faces), colours[name])
    return out


def convert(force: bool = False) -> dict:
    """DAE -> OBJ, STL copied. Returns the provenance record."""
    import trimesh

    OUT.mkdir(exist_ok=True)
    record: dict = {"sources": {}, "files": {}}
    for name, url in SOURCES.items():
        repo = THIRD_PARTY / name
        if not repo.exists():
            raise FileNotFoundError(f"{repo} missing; clone {url} into third_party/")
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        record["sources"][name] = {"url": url, "commit": head}

    todo: list[tuple[Path, str]] = []
    vp = visual_params()
    for link, spec in vp.items():
        for kind in ("visual", "collision"):
            rel = spec[kind]["mesh"]["path"]
            todo.append((THIRD_PARTY / "ur_description" / rel, f"ur_{link}_{kind}"))
    for stem in ("hande", "finger", "finger_collision", "io_coupler"):
        todo.append((THIRD_PARTY / "hande_description" / "meshes" / f"{stem}.dae",
                     f"hande_{stem}"))

    for src, out_stem in todo:
        if not src.exists():
            raise FileNotFoundError(src)
        src_hash = sha256(src)
        if src.suffix.lower() == ".stl":
            dst = OUT / f"{out_stem}.stl"
            if force or not dst.exists():
                shutil.copyfile(src, dst)
            record["files"][out_stem] = {
                "source": str(src.relative_to(THIRD_PARTY)), "source_sha256": src_hash,
                "parts": [{"mesh": dst.name, "rgba": None,
                           "converted_sha256": sha256(dst)}]}
        else:
            parts = []
            for mat, (verts, faces, rgba) in split_by_material(src).items():
                dst = OUT / f"{out_stem}__{mat}.obj"
                if force or not dst.exists():
                    trimesh.Trimesh(vertices=verts, faces=faces,
                                    process=False).export(dst)
                parts.append({"mesh": dst.name, "rgba": list(rgba),
                              "material": mat, "faces": int(len(faces)),
                              "converted_sha256": sha256(dst)})
            record["files"][out_stem] = {
                "source": str(src.relative_to(THIRD_PARTY)),
                "source_sha256": src_hash, "parts": parts}
    (OUT / "PROVENANCE.json").write_text(json.dumps(record, indent=1))
    return record


def main() -> None:
    rec = convert(force=True)
    for name, s in rec["sources"].items():
        print(f"{name:<20} {s['commit'][:12]}  {s['url']}")
    n_parts = sum(len(v["parts"]) for v in rec["files"].values())
    print(f"{len(rec['files'])} source meshes -> {n_parts} parts in {OUT.relative_to(ROOT)}/")
    seen = {}
    for v in rec["files"].values():
        for p in v["parts"]:
            if p["rgba"] and p.get("material"):
                seen[p["material"]] = p["rgba"]
    print("\nmaterials found in the vendor meshes, with their own diffuse RGBA:")
    for m, c in sorted(seen.items()):
        print(f"  {m:<16} {tuple(round(x, 3) for x in c)}")


if __name__ == "__main__":
    main()
