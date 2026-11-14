"""ScriptMind: the stub for test 16's blind pair (plan/test16-live-rule.md, "The blind pair"). Its world is
the stub's, act for act; its acts aimed at the player say one fixed sentence per act kind."""
import json

from asiimov import inhabitant
from asiimov.live import event_dicts
from asiimov.minds import ScriptMind, StubMind
from asiimov.sim import TICKS_PER_DAY, Sim


def play(mind, body=5, days=2, seed=8, agents=14):
    sim = Sim(seed, agents, mind)
    for t in range(days * TICKS_PER_DAY):
        acts = [{"kind": "possess", "agent": body}] if t == 10 else []
        asks = sim.state.get("player", {}).get("asks", [])
        if asks:
            acts.append({"kind": "act", "action": "agree", "target": asks[0]["asker"]})
        elif "player" in sim.state and t % 3 == 0:
            acts.append({"kind": "act", "action": "wait"})
        sim.step(acts)
    return sim


def world(sim):
    """The log as the world, without what minds wrote: mind records, and an intent's goal."""
    out = []
    for e in event_dicts(sim.events):
        if e["kind"] == "genesis" or e["origin"] == "mind":
            continue
        e.pop("mind", None)
        e["data"].pop("goal", None)
        out.append(e)
    return out


def test_the_script_world_is_the_stub_world_and_every_act_aimed_at_the_player_says_a_fixed_sentence():
    stub, script = play(StubMind()), play(ScriptMind())
    assert world(stub) == world(script)
    sent = [inhabitant.scrub(e, 5, script.events) for e in event_dicts(script.events)
            if inhabitant.witnessed(e, 5, set(), -1)]
    aimed = [e for e in sent if e["actor"] not in (5, None) and (
        (e["kind"] == "ask" and e["data"].get("to") == 5) or (e["kind"] == "meet" and e["data"].get("with") == 5)
        or (e["kind"] == "receipt" and e["data"].get("to") == 5 and e["data"].get("action") in inhabitant.SAYS_ACTS
            and len(e["parents"]) < 2))]
    assert aimed, "nothing was aimed at the player"
    for e in aimed:
        if script.events[e["parents"][0]].origin == "player" or script.events[e["parents"][0]].kind == "trade_due":
            continue  # the player's own act, or trade-1's rule buy (no mind reply, so no words)
        assert e["data"].get("says") in ScriptMind.SAYS.values(), e
    said = {e["data"]["says"] for e in sent if "says" in e["data"]}
    assert said <= set(ScriptMind.SAYS.values())
