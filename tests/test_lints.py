"""The AGENTS.md rules as static checks (asiimov/lint.py). The package must be clean, and every rule is
shown failing on a planted violation (a snippet, and the same edit made to the real sim.py) and passing
on a compliant one: a check that has never been seen to fail proves nothing."""
import os
import textwrap

import pytest

from asiimov import lint

SIM = os.path.join(lint.PKG, "sim.py")


def found(src, rule, path="asiimov/sim.py"):
    return [f for f in lint.lint_source(textwrap.dedent(src), path, (rule,))]


def test_the_package_is_clean():
    assert [str(f) for f in lint.lint_package()] == []


def test_the_sim_core_is_what_the_engine_imports():
    """A module the engine comes to import has to be classed as core (checked) or mind side (not)."""
    assert lint.core_imports() == set(lint.SIM_CORE) | set(lint.MIND_SIDE)


def test_each_allowed_index_pick_is_a_live_exception(monkeypatch):
    """No stale entries: without the allowlist, each entry is exactly one finding, and nothing else is."""
    allowed = dict(lint.INDEX_PICK_ALLOW)
    monkeypatch.setattr(lint, "INDEX_PICK_ALLOW", {})
    hits = [f for f in lint.lint_package() if f.rule == "keyed"]
    assert len(hits) == len(allowed)
    assert {(os.path.basename(f.path), f.message.split("`")[1].split("[")[0]) for f in hits} == \
        {(p, name) for p, _, name in allowed}


# ---- int --------------------------------------------------------------------------------------------

INT_BAD = """
import math
from statistics import mean
a = b / c
a /= 2
x = 0.5
y = float(n)
z = math.sqrt(n)
p = n ** -1
q = n ** k
"""
INT_GOOD = """
import math
from math import gcd, isqrt
a = b // c
a //= 2
x = 2 ** 32 + 10 ** 9
y = math.isqrt(n) + gcd(a, b) + isqrt(n)
z = n * 300 // 1000
"""


def test_int_flags_float_arithmetic():
    lines = sorted(f.line for f in found(INT_BAD, "int"))
    assert lines == [3, 4, 5, 6, 7, 8, 9, 10], lines


def test_int_passes_integer_arithmetic():
    assert found(INT_GOOD, "int") == []


# ---- rng --------------------------------------------------------------------------------------------

RNG_BAD = """
import random
from secrets import token_hex
import numpy as np
r = random.random()
s = np.random.rand()
t = time.time()
u = datetime.now()
v = os.urandom(4)
h = hash(name)
i = id(obj)
for x in set(xs):
    pass
ys = [y for y in {1, 2}]
zs = list(set(xs))
fs = os.listdir(d)
"""
RNG_GOOD = """
import hashlib
for x in sorted(set(xs)):
    pass
fs = sorted(f for f in os.listdir(d) if f.endswith(".json"))
r = self.rng.at(f"talk:{aid}", tick, 0, 999)
k = hashlib.sha256(b"x").hexdigest()
seen = {1, 2}
ok = 3 in seen
"""


def test_rng_flags_entropy_clocks_and_unordered_iteration():
    lines = sorted({f.line for f in found(RNG_BAD, "rng")})
    assert lines == [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16], lines


def test_rng_passes_rng_draws_and_sorted_iteration():
    assert found(RNG_GOOD, "rng") == []


# ---- keyed (T22) ------------------------------------------------------------------------------------

KEYED_BAD = """
def talk(self, ids, tick, aid):
    _, roll = self.rng.draw(f"talk:{aid}", 0, 999)
    z = self.rng.normal_milli("mood", 100)
    w = self.rng.draw(make_stream(), 0, 1)
    b = ids[self.rng.at("pick", tick, 0, len(ids) - 1)]
    i = self.rng.at("pick2", tick, 0, len(ids) - 1)
    c = ids[i]
    _, j = self.rng.draw(f"jitter:{aid}", 0, len(ids) - 1)
    d = ids[j]
"""
KEYED_GOOD = """
SYLLABLES = ("ka", "mi")
def talk(self, ids, tick, aid, a):
    idx, jitter = self.rng.draw(f"jitter:{a['id']}", -3, 3)
    stream = f"steal:{a['id']}"
    i1, roll = self.rng.draw(stream, 0, 999)
    hit = self.rng.at(f"talk:{aid}", tick, 0, 999)
    pick_stream = f"pair:{aid}:{tick}"
    b = ids[min((o for o in ids if o != aid), key=lambda o: (self.rng.raw(pick_stream, o), o))]
    s = SYLLABLES[self.rng.draw(f"jitter:{aid}", 0, 1)[1]]
    v = self.rng.draw(f"jitter:{aid}", 0, 1)[1]
"""


def test_keyed_flags_unregistered_counter_draws_and_index_picks():
    lines = sorted(f.line for f in found(KEYED_BAD, "keyed"))
    assert lines == [3, 4, 5, 6, 8, 10], lines


def test_keyed_passes_registered_streams_keyed_draws_and_lowest_score_picks():
    assert found(KEYED_GOOD, "keyed") == []


def test_keyed_leaves_genesis_counter_draws_alone():
    src = 'def gen(rng):\n    age = rng.draw("agent:0", 6, 80)[1]\n'
    assert found(src, "keyed", "asiimov/params.py") == []
    assert len(found(src, "keyed", "asiimov/sim.py")) == 1


# ---- order ------------------------------------------------------------------------------------------

ORDER_BAD = """
class Sim:
    def _tick(self, reqs):
        replies = dispatch(self.mind, reqs, 1)
        r = self.mind.decide(reqs[0])
        if self.order == "shuffle":
            pass
        apply(0, _parse(r))
    def _commit(self):
        for aid in sorted(self.pending):
            apply(0, _parse(replies[aid]))
"""
ORDER_GOOD = """
class Sim:
    def _setup(self, mind, order):
        self.mind, self.order = mind, order
    def _order(self, ids):
        return sorted(ids) if self.order == "id" else ids
    def _commit(self):
        replies = dict(zip(aids, dispatch(self.mind, reqs, self.workers)))
        for aid in self._order(aids):
            apply(eid, _parse(replies[aid]))
"""


def test_order_flags_replies_fetched_or_applied_outside_commit_order():
    lines = sorted(f.line for f in found(ORDER_BAD, "order"))
    assert lines == [4, 4, 5, 5, 6, 8, 11], lines  # line 4: dispatch and self.mind; 5: self.mind and .decide


def test_order_passes_the_commit_shape():
    assert found(ORDER_GOOD, "order") == []


# ---- the same violations planted in the real engine -------------------------------------------------

PLANTED = [
    ("int", "a[\"wake_tick\"] = wake_tick", "a[\"wake_tick\"] = wake_tick / 1"),
    ("rng", "        ids = sorted(ids)\n", "        ids = list(set(ids))\n"),
    ("keyed", "roll = self.rng.at(stream, tick, 0, 999)", "_, roll = self.rng.draw(stream, 0, 999)"),
    ("keyed", "b = agents[min((o for o in ids if o != aid), key=lambda o: (self.rng.raw(pick_stream, o), o))]",
     "b = agents[ids[self.rng.at(pick_stream, tick, 0, len(ids) - 1)]]"),
    ("order", "for aid in self._order(aids):", "for aid in aids:"),
]


@pytest.mark.parametrize("rule,before,after", PLANTED, ids=[f"{r}-{i}" for i, (r, _, _) in enumerate(PLANTED)])
def test_a_violation_planted_in_sim_py_is_flagged(rule, before, after):
    with open(SIM) as f:
        src = f.read()
    assert src.count(before) == 1, before
    rules = ("int", "rng", "keyed", "order")
    assert lint.lint_source(src, "asiimov/sim.py", rules) == []
    hits = lint.lint_source(src.replace(before, after), "asiimov/sim.py", rules)
    assert hits and {f.rule for f in hits} == {rule}, hits
