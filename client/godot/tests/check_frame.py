"""Screenshot smoke check for the V-pass (Steering 2026-11-16 08:53Z): the inhabitant frame must draw the map
with its art and the state, plan and talk panels, not merely a non-blank frame.

    python client/godot/tests/check_frame.py SHOT.png

Regions are the client's fixed layout at its default 1280x800 window: the map at the left, "What's happening"
and "Said near you" in the middle, "You" / "Today's plan" / "Here with you" at the right."""
import sys

from PIL import Image


def _counts(img, box):
    return img.crop(box).getcolors(1 << 24) or []


def text_pixels(img, box) -> int:
    """Light pixels (the panels' text, white or pale blue) inside box: well above the grey background."""
    return sum(n for n, (r, g, b) in _counts(img, box) if (r + g + b) / 3 > 140)


def colours(img, box) -> int:
    return len(_counts(img, box))


def main(path: str) -> int:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    fails = []
    if colours(img, (0, 0, 340, 580)) < 40:
        fails.append("the map has no art (too few colours)")
    greens = sum(n for n, (r, g, b) in _counts(img, (0, 0, 340, 580)) if g > r + 30 and g > b + 30)
    if greens < 20000:
        fails.append(f"the map has no grass ({greens} green pixels)")
    for name, box in {"state (You)": (880, 0, w, 180), "plan (Today's plan)": (880, 180, w, 240),
                      "feed (What's happening)": (340, 0, 870, 60), "talk (Said near you)": (340, 480, 870, 540)}.items():
        n = text_pixels(img, box)
        if n < 60:
            fails.append(f"no {name} panel text ({n} light pixels)")
    for f in fails:
        print("FAIL:", f)
    print("frame ok" if not fails else f"{len(fails)} checks failed", f"({w}x{h})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
