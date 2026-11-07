"""Replay viewer (asiimov/viewer.py): who-is-where read from the log matches the world, tick by tick.

The viewer never replays; it reads places from wake, activity and sleep events. That is only right if
every place change is emitted (AGENTS.md: no silent state edits), so this checks it against the live sim.
"""
import gzip
import json
import os
import shutil
import subprocess

import pytest

from asiimov.minds import StubMind
from asiimov.sim import TICKS_PER_DAY, Sim
from asiimov.viewer import build, main, moves, read_log, render

FIRE = {TICKS_PER_DAY + 60: [{"kind": "fire", "family": 2}]}


@pytest.fixture(scope="module")
def world():
    """A 30-agent, 3-day stub world with a fire, and each villager's (place, activity) after every tick."""
    sim = Sim(7, 30, StubMind())
    truth = []
    for tick in range(3 * TICKS_PER_DAY):
        sim.step(FIRE.get(tick, []))
        truth.append([(a["place"], a["activity"]) for a in sim.state["agents"]])
    sim.finish()
    return [json.loads(e.to_json()) for e in sim.events], truth


def at(m, tick):
    return next(x for x in reversed(m) if x[0] <= tick)[1:3]


def test_places_and_activities_from_the_log_match_the_world_every_tick(world):
    events, truth = world
    mv = moves(events)
    for tick, row in enumerate(truth):
        for aid, (place, activity) in enumerate(row):
            assert at(mv[aid], tick) == [place, activity], (tick, aid)
    # the check covered the cases that matter: visits to a person, the public places, and sleep at home
    seen = {(p, a) for row in truth for p, a in row}
    assert "visit" in {a for _, a in seen} and "sleep" in {a for _, a in seen}
    assert {"market", "field", "plaza", None} <= {p for p, _ in seen}  # None: walking


def test_page_embeds_the_log_and_cant_be_broken_by_text_in_it(world):
    events, _ = world
    events = json.loads(json.dumps(events))
    m = next(e for e in events if e["origin"] == "mind")
    m["mind"]["raw_output"] = '{"reason":"</script><script>alert(1)</script>"}'
    html = render(build(events, "test"))
    assert html.count("</script>") == html.count("<script")  # the mind's text didn't close a tag
    blob = html.split('<script id="view" type="application/json">', 1)[1].split("</script>", 1)[0]
    view = json.loads(blob)
    assert len(view["events"]) == len(events) and view["complete"] and view["end_tick"] == 3 * TICKS_PER_DAY
    assert "population" not in view["events"][0][5]  # carried once, as view["agents"]


def test_a_cut_off_gzipped_log_still_opens(world, tmp_path):
    events, _ = world
    lines = [json.dumps(e) for e in events[:-1]]  # no end event, as when a run crashed
    path = tmp_path / "events.jsonl.gz"
    with gzip.open(path, "wt") as f:
        f.write("\n".join(lines) + "\n" + lines[-1][:40])  # and a half-written last line
    got, bad = read_log(str(tmp_path))
    assert bad == 1 and len(got) == len(events) - 1
    main([str(path), "--out", str(tmp_path / "v.html")])
    assert "application/json" in (tmp_path / "v.html").read_text()
    view = build(got, "x", bad)
    assert not view["complete"] and view["end_tick"] == got[-1]["tick"] + 1


# Mind outputs the engine would reject are still recorded verbatim, and the page reads raw_output, so it must
# survive any shape (PR #24 review: an <img onerror> in a plan's start_h ran script; a string "then" or "core"
# threw and blanked every pane listing it).
HOSTILE = {
    "mind_plan": '{"plan":[{"activity":"work","start_h":"<img src=x onerror=window.pwned=1>","end_h":9},"x",null]}',
    "mind_deliberate": '{"action":"<img src=x onerror=window.pwned=2>","target":"0","then":"shop","goal":{"a":1},"reason":7}',
    "mind_reflect": '{"core":"<img src=x onerror=window.pwned=3>","summary":["<b>x</b>"]}',
}
BROWSER = """
const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch(); const p = await b.newPage(); const errs = [];
  p.on('pageerror', e => errs.push(String(e)));
  await p.goto('file://' + process.argv[2]);
  await p.click('.tabs button[data-tab="feed"]'); await p.selectOption('#win', {index: 3});
  const rows = await p.locator('#pane .ev').count();
  await p.locator('#pane .ev').first().click();
  await p.evaluate(a => { S.agent = a; S.tab = "agent"; drawPane(); }, +process.argv[3]);
  await p.waitForTimeout(200);
  console.log(JSON.stringify({rows, errs, pwned: await p.evaluate(() => window.pwned || null),
    imgs: await p.locator('#pane img').count(), timeline: await p.locator('#tl .ev').count()}));
  await b.close();
})();
"""


def test_hostile_mind_output_is_shown_as_text_and_breaks_no_pane(world, tmp_path):
    node = shutil.which("node")
    root = subprocess.run(["npm", "root", "-g"], capture_output=True, text=True).stdout.strip() if node else ""
    if not node or not os.path.isdir(f"{root}/playwright"):
        pytest.skip("needs node and a global playwright (not installed in CI)")
    events = json.loads(json.dumps(world[0]))
    for kind, raw in HOSTILE.items():  # the first output of each kind
        hit = next(e for e in events if e["kind"] == kind)
        hit["mind"]["raw_output"] = raw
    page, script = tmp_path / "v.html", tmp_path / "check.js"
    page.write_text(render(build(events, "hostile")))
    script.write_text(BROWSER)
    out = subprocess.run([node, str(script), str(page), str(hit["actor"])], capture_output=True, text=True, timeout=120,
                         env={**os.environ, "NODE_PATH": root})
    assert out.stdout.strip(), out.stderr[-2000:]  # the page broke before the checks could run
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got["errs"] == [] and got["pwned"] is None and got["imgs"] == 0, got
    assert got["rows"] > 100 and got["timeline"] > 0, got
