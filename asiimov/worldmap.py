"""The village map: where each place is, and how long walking between two points takes.

Integer tiles of TILE_M metres; the plaza's anchor is (0, 0). Every place has a rectangle (x, y, w, h)
for drawing and an anchor tile for distance. Distance is Manhattan along an L path (x first, then y),
nothing blocks it, and travel takes ceil(distance / WALK) ticks. The layout is a function of the number
of families alone, with no randomness, and genesis records it, so a replay checks it like any event.
Prompts keep naming places; coordinates are for the engine and the client (plan/t6-map-spec.md).
"""
TILE_M = 20
WALK = 40          # tiles per tick: 800 m per 10 minutes, about 4.8 km/h
RING0 = 20         # the first ring of home lots, in tiles from the plaza's anchor
RING_STEP = 8      # rings and lots are this many tiles apart
EDGE = 12          # field and woods sit this far beyond the first ring with no homes


def _rect(anchor, w, h):
    return [anchor[0] - w // 2, anchor[1] - h // 2, w, h]


def _ring(d: int) -> list[tuple[int, int]]:
    """Lot anchors on the square ring at Chebyshev distance d, clockwise from its top-left corner."""
    s = range(-d, d, RING_STEP)
    return ([(x, -d) for x in s] + [(d, y) for y in s] + [(-x, d) for x in s] + [(-d, -y) for y in s])


def build_map(homes: list[str]) -> dict:
    """The layout for these homes, which take lots in the order given (family id order)."""
    lots, d = [], RING0
    while len(lots) < len(homes):
        lots += _ring(d)[:len(homes) - len(lots)]
        d += RING_STEP
    edge = d + EDGE  # d is now the first ring with no homes
    places = {"plaza": (0, 0, 10, 10), "market": (14, 0, 8, 6), "workshop": (-14, 0, 8, 6),
              "field": (0, edge, 30, 16), "woods": (0, -edge, 40, 16)}
    out = {name: {"anchor": [x, y], "rect": _rect((x, y), w, h)} for name, (x, y, w, h) in places.items()}
    for home, (x, y) in zip(homes, lots):
        out[home] = {"anchor": [x, y], "rect": _rect((x, y), 6, 6)}
    return {"tile_m": TILE_M, "walk": WALK, "places": out}


def distance(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def travel_ticks(a, b) -> int:
    return -(-distance(a, b) // WALK)


def position(route: dict, anchor_to, tick: int) -> list[int]:
    """Where a walker on `route` is at `tick`: WALK tiles per tick along the L path, x first."""
    (x, y), (tx, ty) = route["from"], anchor_to
    left = WALK * max(0, tick - route["depart"])
    dx = min(left, abs(tx - x))
    x += dx if tx >= x else -dx
    dy = min(left - dx, abs(ty - y))
    y += dy if ty >= y else -dy
    return [x, y]
