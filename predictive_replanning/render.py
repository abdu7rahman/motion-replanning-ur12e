"""Turn a rendered run into a GIF, with the measured state burned in.

Rendering is MuJoCo's own, offscreen through OSMesa. That needs libosmesa6,
which is a *system* package -- pip cannot supply it, and without it MuJoCo
fails inside PyOpenGL with `'NoneType' object has no attribute 'glGetError'`,
an error that names neither MuJoCo nor the missing library. On a bare Ubuntu
image `apt-get update` has to run first or the install simply does not find it.

The overlay is drawn after the fact from the same numbers the trial reports,
so what the caption says and what the table says cannot drift apart.
"""
from __future__ import annotations

__author__ = "".join(
    chr(c - 7) for c in (104, 105, 107, 124, 115, 39, 121, 104, 111, 116, 104, 117)
)

from PIL import Image, ImageDraw

INK = (29, 29, 31)
MUTED = (110, 110, 115)
SIGNAL = (215, 0, 21)


def make_gif(frames, path: str, *, title: str, fps: int = 20, scale: float = 1.0) -> str:
    imgs = []
    for fr in frames:
        im = Image.fromarray(fr["px"])
        if scale != 1.0:
            im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, im.width, 46], fill=(251, 251, 253))
        d.text((10, 6), title, fill=INK)
        hit = fr["hit"]
        d.text((10, 26),
               f"t {fr['t']:4.2f}s   clearance {fr['clearance']:.3f} m   replans {fr['replans']}",
               fill=SIGNAL if hit else MUTED)
        if hit:
            d.rectangle([0, 0, im.width - 1, im.height - 1], outline=SIGNAL, width=4)
        imgs.append(im.convert("P", palette=Image.ADAPTIVE, colors=128))
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0, optimize=True)
    return path
