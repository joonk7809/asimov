"""The client's pixel art (V-pass, joon 2026-11-16: "cute design features ... help with immersion").

Original 16 px sprites drawn here as character grids, so every pixel is ours (no third-party art, nothing
to license). Writes client/godot/art/*.png; the client loads them at runtime (Image.load_from_file), so they
need no editor import. Run from the repo root:  python client/godot/tools/make_art.py

White and grey pixels in a "tint" layer are meant to be coloured in the client with modulate (a roof by
family, a tunic by role); everything else keeps its own colour."""
import os
import random

from PIL import Image

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "art")

PAL = {
    ".": None,
    "k": (40, 32, 42), "K": (70, 58, 66),          # outline, soft outline
    "w": (255, 255, 255), "W": (210, 210, 214),    # tint white, tint shade
    "s": (247, 205, 168), "S": (222, 170, 134),    # skin, skin shade
    "h": (92, 60, 44), "H": (140, 94, 60),         # hair, hair light
    "e": (34, 30, 40), "p": (238, 136, 150),       # eye, cheek
    "b": (122, 82, 56), "B": (88, 58, 40),         # wood, wood dark
    "r": (214, 84, 76), "R": (170, 58, 56),        # red, red dark
    "c": (250, 244, 228), "C": (218, 208, 188),    # cream, cream shade
    "g": (112, 178, 92), "G": (82, 146, 70), "f": (150, 204, 110),  # greens
    "t": (60, 120, 66), "T": (44, 92, 54),         # tree greens
    "d": (196, 160, 110), "D": (168, 132, 88),     # dirt
    "y": (246, 214, 92), "Y": (222, 176, 60),      # yellow
    "a": (160, 164, 176), "A": (120, 124, 138),    # stone
    "u": (110, 170, 222), "U": (78, 132, 190),     # water
    "o": (255, 186, 214),                          # flower pink
    "z": (200, 214, 255),
}

SPRITES = {
    # A villager: the tunic (w/W) is tinted by role in the client; head, hair and boots keep their colours.
    "villager": """
................
......kkkk......
.....khhhhk.....
....khHhhHhk....
....khssssshk...
....ksesseesk...
....kpssssspk...
.....kssSssk....
....kkwwwwwkk...
...kswwwwwwwsk..
...kswWwwwWwsk..
....kwwwwwwwk...
....kwWwwwWwk...
.....kBk.kBk....
.....kkk.kkk....
................""",
    # A house: its roof (w/W) is tinted per family in the client.
    "house_roof": """
................
.......kk.......
......kwwk......
.....kwwWwk.....
....kwwWWwwk....
...kwwWwwWwwk...
..kwwWwwwwWwwk..
.kkkkkkkkkkkkkk.
................
................
................
................
................
................
................
................""",
    "house_base": """
................
................
................
................
................
................
................
................
..kccccccccccck.
..kcCkkcccbbbck.
..kcCuukccbBbck.
..kcCkkcccbBbck.
..kcCcccccbBbck.
..kcCcccccbBbck.
..kkkkkkkkkkkkk.
................""",
    "stall": """
................
..kkkkkkkkkkkk..
.krrcrrcrrcrrck.
.krrcrrcrrcrrck.
.kRkRkRkRkRkRkk.
..kb........bk..
..kb..y.r.g.bk..
..kb.yYrRgG.bk..
..kbbbbbbbbbbk..
..kBccccccccBk..
..kBccccccccBk..
..kBkkkkkkkkBk..
..kB........Bk..
..kk........kk..
................
................""",
    "workshop": """
................
..........kk....
..........kAk...
...kkkkkkkkAkk..
..kBBBBBBBBBBBk.
.kBbbbbbbbbbbbBk
.kkkkkkkkkkkkkkk
..kccccccccccck.
..kcBBBBcaaaack.
..kcBkkBckAAkck.
..kcBkkBcaaaack.
..kcBkkBccAAcck.
..kcBkkBcAAAack.
..kkkkkkkkkkkkk.
................
................""",
    "well": """
................
.....kkkkkk.....
....kBbbbbBk....
....kb....bk....
....kb....bk....
....kb.kk.bk....
....kb.kB.bk....
...kaaaaaaaak...
..kaAaaAaaAaak..
..kauuuuuuuuak..
..kaUuUuuUuUak..
..kaAaaAaaAaak..
...kaaaaaaaak...
....kkkkkkkk....
................
................""",
    "tree": """
................
......kkkk......
....kktttTkk....
...ktttfttTTk...
..kttfftttTTTk..
..ktffttttTTTk..
..kttttttTTTTk..
..kTttttTTTTTk..
...kTTttTTTTk...
....kkTTTTkk....
......kbBk......
......kbBk......
.....kbbBBk.....
......kkkk......
................
................""",
    "crops": """
................
.g.g.g.g.g.g.g.g
gGgGgGgGgGgGgGgG
dDdDdDdDdDdDdDdD
dddddddddddddddd
.y.y.y.y.y.y.y.y
gYgYgYgYgYgYgYgY
dDdDdDdDdDdDdDdD
dddddddddddddddd
.g.g.g.g.g.g.g.g
gGgGgGgGgGgGgGgG
dDdDdDdDdDdDdDdD
dddddddddddddddd
.y.y.y.y.y.y.y.y
gYgYgYgYgYgYgYgY
dDdDdDdDdDdDdDdD""",
    "marker": """
....kkkkkk......
...kyyyyyyk.....
...kyYyyYyk.....
....kyyyyk......
.....kyyk.......
......kk........
................
................""",
}


def grid(name: str) -> Image.Image:
    rows = [r for r in SPRITES[name].strip("\n").split("\n")]
    w = max(len(r) for r in rows)
    img = Image.new("RGBA", (16 if w <= 16 else w, len(rows)), (0, 0, 0, 0))
    for y, row in enumerate(rows):
        for x, ch in enumerate(row[:16]):
            col = PAL.get(ch)
            if col:
                img.putpixel((x, y), col + (255,))
    return img


def noise_tile(seed: str, base, dots) -> Image.Image:
    """A 16 px ground tile: one colour with a deterministic scatter of others (tileable: no edges)."""
    rng = random.Random(seed)
    img = Image.new("RGBA", (16, 16), PAL[base] + (255,))
    for ch, n in dots:
        for _ in range(n):
            img.putpixel((rng.randrange(16), rng.randrange(16)), PAL[ch] + (255,))
    return img


def cobble() -> Image.Image:
    img = Image.new("RGBA", (16, 16), PAL["a"] + (255,))
    for y in range(16):
        for x in range(16):
            if y % 4 == 3 or (x + (4 if (y // 4) % 2 else 0)) % 8 == 7:
                img.putpixel((x, y), PAL["A"] + (255,))
    return img


def main():
    os.makedirs(OUT, exist_ok=True)
    tiles = {
        "grass": noise_tile("grass", "g", [("G", 22), ("f", 10), ("o", 2), ("y", 1)]),
        "dirt": noise_tile("dirt", "d", [("D", 26), ("C", 4)]),
        "woodsfloor": noise_tile("woods", "G", [("T", 30), ("g", 10)]),
        "cobble": cobble(),
    }
    for name, img in tiles.items():
        img.save(os.path.join(OUT, name + ".png"))
    for name in SPRITES:
        if name != "villager":  # written only as its two layers below
            grid(name).save(os.path.join(OUT, name + ".png"))
    # The villager in two layers, so the client tints only the tunic by role: its w/W pixels, and the rest.
    full = grid("villager")
    tint, base = Image.new("RGBA", full.size), Image.new("RGBA", full.size)
    for y in range(full.height):
        for x in range(full.width):
            px = full.getpixel((x, y))
            (tint if px[:3] in (PAL["w"], PAL["W"]) else base).putpixel((x, y), px)
    tint.save(os.path.join(OUT, "villager_tint.png"))
    base.save(os.path.join(OUT, "villager_base.png"))
    print(f"wrote {len(tiles) + len(SPRITES) + 1} images to {OUT}")


if __name__ == "__main__":
    main()
