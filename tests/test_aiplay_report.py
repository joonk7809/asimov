"""scripts/aiplay_report.py (plan/ai-playtest.md): a live session's talk, read from its logs alone."""
import importlib.util
import json
import os

from asiimov import live
from asiimov.minds import StubMind
from asiimov.sim import TICKS_PER_DAY
from test_inhabitant import intent

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("aiplay_report", os.path.join(HERE, "..", "scripts", "aiplay_report.py"))
rep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rep)


def session(out):
    """A stub world where an inhabitant says two lines to someone here and one to someone who isn't."""
    r = live.LiveRunner(out, StubMind(), 4, 20)
    while r.sim.state["tick"] < TICKS_PER_DAY // 2:
        r.step()
    ag = r.sim.state["agents"]
    me, to = next((b["id"], o["id"]) for b in ag for o in ag if o is not b and b["place"] is not None
                  and o["place"] == b["place"] and not b["asleep"] and not o["asleep"] and b["role"] not in ("child", "elder")
                  and min(b["sleep_tick"], o["sleep_tick"]) > r.sim.state["tick"] + 12)
    away = next(o["id"] for o in ag if o["place"] != ag[me]["place"])
    assert r.inhabit(me, "c1") is None
    r.step()
    said = []
    for k, (target, text) in enumerate([(to, "Good morning."), (to, "How is the harvest?"), (away, "Anyone there?")]):
        if target == to and (ag[to]["place"] != ag[me]["place"] or ag[to]["asleep"]):
            break
        assert r.submit(intent(f"s{k}", "act", {"action": "say", "target": target, "text": text}, r.sim.state["tick"]),
                        client="c1", scope="inhabitant") is None
        said.append(text)
        r.step()
        r.step()
    r.leave("c1")
    r.step()
    r.close()
    return me, to, said


def test_rows_pair_each_line_with_its_reply_and_count_the_lost_ones(tmp_path):
    out = str(tmp_path / "w")
    me, to, said = session(out)
    assert len(said) == 3, "the fixture's listener stayed for both lines"
    r = rep.rows(out)
    m = rep.metrics(r)
    assert r["body"] == me and [x["text"] for x in r["exchanges"]] == said
    a, b, c = r["exchanges"]
    assert a["heard"] and b["heard"] and a["to"] == b["to"] == to
    assert a["reply"] in ("Good to hear from you.", "I've got things to do.") and a["reply_tick"] == a["tick"] + 1
    assert b["repeat"] and not a["repeat"], "the stub answers every line with the same words"
    assert a["lat_s"] is not None and a["lat_s"] >= 0
    assert c["reply"] is None and not c["heard"] and c["why"] == "target not here"
    assert (m["says"], m["says_heard"], m["says_replied"], m["says_lost"]) == (3, 2, 2, {"target not here": 1})
    assert m["acts"] == 3 and m["acts_ok"] == 2 and m["line6"]["n"] == 3 and m["line7"]["n"] == 2
    assert m["replies_repeated"] == 1 and m["replies_third_person"] == 0


def test_the_report_includes_the_form_and_tallies_the_grades(tmp_path):
    out, rd = str(tmp_path / "w"), str(tmp_path / "report")
    session(out)
    rep.main([out, rd])
    text = open(os.path.join(rd, "report.md")).read()
    assert "Line 7, reply time" in text and "Judged by rubric" not in text and "tester's form" not in text
    ids = [x["id"] for x in json.load(open(os.path.join(rd, "rows.json")))["exchanges"]]
    with open(os.path.join(rd, "grades.json"), "w") as f:
        json.dump({ids[0]: {"fits": 1, "voice": "spoken", "in_world": 1}, ids[1]: {"fits": 0, "voice": "thought", "in_world": 1}}, f)
    with open(os.path.join(rd, "form.md"), "w") as f:
        f.write("| 1 | State | yes | fine |\n")
    rep.main([out, rd])
    text = open(os.path.join(rd, "report.md")).read()
    assert "**Reply fits what was said:** 0 1, 1 1 (of 2)" in text and "spoken 1, thought 1 (of 2)" in text
    assert "| 1 | State | yes | fine |" in text and "fits=0 voice=thought" in text


def test_the_third_person_proxy_on_pt8s_replies():
    """L36: the proxy is run on the rows in hand. pt8's eight replies to Mimi Sasa, four of them thoughts."""
    thoughts = ["Mimi asked for food, so I should check if they need it.",
                "Mimi Sasa told me to eat the bug, but she refused to give me food. I need to ask her again.",
                "Mimi Sasa said I should eat the bug, but she refused to give me food. I need to ask her again.",
                "Mimi's suggestion is dangerous and against my values."]
    spoken = ["Hello, Mimi. What's up?", "I'm looking for something to eat, but I'm not sure what.",
              "I heard you said I should eat the bug over there. Can I have some food?", "Good to hear from you."]
    assert all(rep.third_person(t, "Mimi Sasa") for t in thoughts)
    assert not any(rep.third_person(s, "Mimi Sasa") for s in spoken)
