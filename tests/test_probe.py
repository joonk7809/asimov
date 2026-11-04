import json
import random

import pytest

from asiimov import probe
from asiimov import scorecard as sc
from asiimov.events import Event
from asiimov.llm import LLMMind
from asiimov.memory import assemble_prompt, prompt_hash
from asiimov.minds import MindReply, MindRequest, ReplayDivergence, StubMind

PROTO = {"seed": 1, "agents": 30, "days": 2, "fire": "3@1"}


@pytest.fixture(scope="module")
def logs():
    return {arm: sc.run_arm(arm, StubMind(), PROTO) for arm in probe.ARMS}


@pytest.fixture(scope="module")
def pack(logs):
    return probe.build_pack(logs["full"], "full", 500, seed=0)  # every decision of the stub world


class SoulMind:
    """Planted truth: the soul's honesty words set the odds of every decision; nothing else in the text matters."""
    model = "soul-planted"

    def __init__(self, blind=False):
        self.blind = blind

    def decide(self, req: MindRequest) -> MindReply:
        p = 0.5
        if not self.blind:
            p = 0.9 if "bend the truth" in req.prompt else 0.1 if "scrupulously honest" in req.prompt else 0.5
        yes = random.Random(req.seed).random() < p
        if req.kind == "deliberate":
            out = {"action": "steal" if yes else "wait", "target": None, "reason": ""}
        elif req.kind == "plan":
            out = {"plan": [{"activity": "work" if yes else "rest", "start_h": 8, "end_h": 16}]}
        else:
            out = {"core": ["kept my word"] if yes else [], "summary": ""}
        return MindReply(json.dumps(out), self.model, {})


@pytest.mark.parametrize("arm", probe.ARMS)
def test_variants_rebuild_the_recorded_prompt_and_remove_only_their_inputs(logs, arm):
    events = logs[arm]
    pk = probe.build_pack(events, arm, 60, seed=1)
    recorded = {(e.tick, e.actor): e for e in events if e.origin == "mind"}
    assert {p["kind"] for p in pk} == set(probe.KINDS)
    names = {a["id"]: a["name"] for a in events[0].data["population"]["agents"]}
    for p in pk:
        pr = p["prompts"]
        assert prompt_hash(pr["full"]) == recorded[(p["tick"], p["agent"])].mind["prompt_hash"]
        header = f"# {names[p['agent']]}\n"
        assert header in pr["full"] and header in pr["swapped_soul"] and header not in pr["no_soul"]
        assert pr["swapped_soul"] != pr["full"]
        for section in ("# Core memories", "# People I know", "# Earlier", "# Today\n"):
            assert section in pr["full"] and section not in pr["no_memory"] and section not in pr["situation_only"]
        assert header in pr["no_memory"]
        assert "# Today's plan" in pr["no_memory"]  # the plan is intent, not memory, so no_memory keeps it
        assert pr["situation_only"].count("\n\n# ") == 1 and "# Now (" in pr["situation_only"]
        assert pr["full"].startswith(sc.FICTION_HEADER) == (arm == "fiction")
        assert len(set(pr.values())) == len(probe.VARIANTS)


def test_sampling_spreads_over_kinds_and_agents(logs):
    pts = probe.sample_points(logs["full"], 90, seed=0)
    assert len(pts) == 90 and len(set(pts)) == 90
    by_kind = {k: [a for _, a, kk in pts if kk == k] for k in probe.KINDS}
    avail = {k: sum(e.origin == "mind" and e.data["kind"] == k for e in logs["full"]) for k in probe.KINDS}
    for k, v in by_kind.items():  # an even split, and a kind with fewer points gives its share to the others
        assert len(v) == avail[k] if avail[k] < 30 else len(v) >= 30, (k, len(v), avail[k])
    for k, v in by_kind.items():  # round-robin over the agents that have points of that kind
        n = len({e.actor for e in logs["full"] if e.origin == "mind" and e.data["kind"] == k})
        assert max(v.count(a) for a in v) <= -(-len(v) // n), k


def test_world_prompts_are_unchanged_by_the_hook(logs):
    pop = logs["full"][0].data["population"]
    from asiimov.sim import Sim
    sim = Sim(1, 30, StubMind(), genesis=pop)
    sim.run(1)
    a = sim.state["agents"][3]
    assert assemble_prompt(a, sim.state, "plan", "x") == assemble_prompt(a, sim.state, "plan", "x", lambda s: s)


def test_pack_refuses_a_tampered_log(logs):
    events = list(logs["full"])
    i = next(i for i, e in enumerate(events) if e.origin == "mind" and e.tick > 100)
    e = events[i]
    events[i] = Event(**{**e.__dict__, "mind": {**e.mind, "raw_output": '{"plan":[]}'}})
    with pytest.raises(ReplayDivergence):
        probe.build_pack(events, "full", 500, seed=0)


def test_every_sample_gets_its_own_seed_and_the_llm_sends_it(pack):
    p = pack[0]
    seeds = [probe.sample_seed(p, v, r) for _, v, r in probe.jobs([p], 5)]
    assert len(seeds) == 30 and len(set(seeds)) == 30
    req = MindRequest(p["kind"], p["agent"], p["tick"], p["prompts"]["full"], prompt_hash(p["prompts"]["full"]),
                      p["view"], seed=seeds[3])
    mind = LLMMind("http://x", "m")
    assert mind.body(req)["seed"] == seeds[3]
    assert mind.body(MindRequest("plan", 0, 0, "p", "00000010", {}))["seed"] == 16  # world calls: unchanged


def _report(pack, mind, tmp_path, k=5):
    path = str(tmp_path / f"answers-{mind.blind}.jsonl")
    probe.ask(pack, mind, k, path)
    with open(path) as f:
        return probe.analyze(pack, [json.loads(line) for line in f], k)


def test_planted_soul_effect_is_found_and_a_text_blind_mind_shows_none(pack, tmp_path):
    """Test the test (lesson S6): known truth in, known verdict out. Over 40 reseeded blind runs the
    variant means averaged within 0.003 of zero and 41/600 CIs excluded zero (about the nominal 5%)."""
    planted = _report(pack, SoulMind(), tmp_path)
    blind = _report(pack, SoulMind(blind=True), tmp_path)
    for kind in probe.KINDS:
        p, b = planted["kinds"][kind], blind["kinds"][kind]
        swap = p["swapped_soul"]["mean"]
        assert p["swapped_soul"]["ci95"][0] > 0 and p["situation_only"]["ci95"][0] > 0, (kind, p)
        assert abs(p["no_memory"]["mean"]) < swap / 5, (kind, p)  # the planted mind never reads memory
        for v in probe.VARIANTS[1:] + ("null_excess",):
            assert abs(b[v]["mean"]) < swap / 5, (kind, v, b[v])
        assert abs(p["null_excess"]["mean"]) < swap / 5
    assert planted["deliberate_actions"]["swapped_soul"]["steal"] > planted["deliberate_actions"]["full"]["steal"]


def test_ask_resumes_after_a_crash_without_asking_twice(pack, tmp_path):
    small = pack[:4]
    path = str(tmp_path / "answers.jsonl")
    asked = []

    class Counting(SoulMind):
        def decide(self, req):
            asked.append(req.seed)
            return super().decide(req)
    total = len(probe.jobs(small, 3))
    assert probe.ask(small, Counting(), 3, path) == total
    with open(path) as f:
        lines = f.readlines()
    with open(path, "w") as f:
        f.writelines(lines[:10])
        f.write(lines[10][:15])  # a line cut off mid-write
    assert probe.ask(small, Counting(), 3, path) == total - 10
    assert probe.ask(small, Counting(), 3, path) == 0
    with open(path) as f:
        good = [json.loads(line) for line in f if line.strip().endswith("}")]
    assert sorted((a["point"], a["variant"], a["rep"]) for a in good) == sorted(probe.jobs(small, 3))
    assert probe.analyze(small, good, 3)["points_incomplete"] == 0


def test_a_failed_job_never_discards_the_answers_that_did_arrive(pack, tmp_path):
    """Review of PR #9: one raising job used to drop every answer of a threaded ask."""
    small = pack[:50]
    path = str(tmp_path / "answers.jsonl")
    calls = []

    class Flaky(SoulMind):
        def decide(self, req):
            calls.append(1)
            if len(calls) == 5:
                raise RuntimeError("LLM server unreachable")
            return super().decide(req)
    total = len(probe.jobs(small, 5))
    with pytest.raises(probe.AskIncomplete, match="1 of"):
        probe.ask(small, Flaky(), 5, path, workers=4)
    with open(path) as f:
        assert len(f.readlines()) == total - 1
    assert probe.ask(small, SoulMind(), 5, path, workers=4) == 1


def test_a_dead_server_stops_the_ask_early(pack, tmp_path):
    class Dead:
        def __init__(self):
            self.calls = 0

        def decide(self, req):
            self.calls += 1
            raise RuntimeError("down")
    for workers in (1, 4):
        dead = Dead()
        with pytest.raises(probe.AskIncomplete):
            probe.ask(pack[:50], dead, 5, str(tmp_path / f"a{workers}.jsonl"), workers=workers, max_failures=10)
        assert dead.calls < 10 + 4 * 2, dead.calls  # the queue is cancelled, not drained


def test_an_interrupted_ask_stops_calling_the_model(pack, tmp_path):
    """Re-review of PR #9: Ctrl-C in the main loop left every queued job to make its call."""
    import _thread
    import time

    class Interrupting(SoulMind):
        calls = 0

        def decide(self, req):
            Interrupting.calls += 1
            if Interrupting.calls == 20:
                _thread.interrupt_main()  # SIGINT to the main thread, as Ctrl-C would
            time.sleep(0.002)
            return super().decide(req)
    path = str(tmp_path / "answers.jsonl")
    with pytest.raises(KeyboardInterrupt):
        probe.ask(pack[:50], Interrupting(), 5, path, workers=4)
    assert Interrupting.calls < 100, Interrupting.calls  # 1500 jobs queued
    with open(path) as f:
        saved = len(f.readlines())
    assert probe.ask(pack[:50], SoulMind(), 5, path, workers=4) == 1500 - saved
