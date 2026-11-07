"""asiimov.cost (load and price) and the bench's sampling, with a stub world and a fake server."""
import json
import sys
import threading
from http.server import ThreadingHTTPServer

from asiimov import bench, cost
from asiimov.events import write_jsonl
from asiimov.minds import StubMind
from asiimov.sim import TICKS_PER_DAY, Sim
from tests.test_llm import SEEN, Fake


def test_load_counts_every_mind_call_per_agent_day_and_per_tick(tmp_path):
    sim = Sim(1, 30, StubMind())
    sim.run(2)
    path = tmp_path / "events.jsonl"
    write_jsonl(sim.events, str(path))
    out = cost.load(str(path))
    minds = [e for e in sim.events if e.origin == "mind"]
    ticks = sim.events[-1].tick + 1
    assert out["agents"] == 30 and out["calls"] == len(minds)
    assert out["calls_per_agent_day"] == round(len(minds) / 30 / (ticks / TICKS_PER_DAY), 3)
    assert out["by_kind_per_agent_day"]["plan"] > 0
    per_tick = max(sum(1 for e in minds if e.tick == t) for t in range(ticks))
    assert out["calls_per_tick_per_100_agents"]["max"] == round(per_tick * 100 / 30, 3)
    assert abs(sum(out["share_by_hour"].values()) - 1) < 0.02


def test_price_turns_throughput_and_load_into_dollars():
    loads = [{"calls": 480, "agents": 100, "days": 1.0,
              "calls_per_tick_per_100_agents": {"mean": 2.0, "p99": 8.0}}]
    benches = [{"model": "m", "gpu": "g", "cost_per_hr": 3.6,
                "results": [{"req_per_s": 1.0, "concurrency": 1}, {"req_per_s": 10.0, "concurrency": 64}]}]
    out = cost.price(loads, benches, [1440], [100])
    s = out["servers"][0]
    assert out["calls_per_agent_day"] == 4.8 and out["peak_to_mean_p99"] == 4.0
    assert s["usd_per_1k_calls"] == 0.1  # $3.6/h over 36,000 calls/h
    # 4.8 calls per agent-day = 0.2 per agent world-hour, at $0.0001 each
    assert s["usd_per_agent_world_hour"] == 0.00002
    g = s["grid"][0]  # a world day per real hour: 24 world hours per real hour
    assert g["usd_per_player_hour"] == round(0.00002 * 24 * 100, 5)
    assert g["agents_per_gpu"] == int(10 / (4.8 / 3600))  # 4.8 calls per agent per real hour


def test_bench_never_sends_a_prompt_twice_and_cache_bust_makes_every_prefix_unique(tmp_path, monkeypatch):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    SEEN.clear()
    out = tmp_path / "b.json"
    monkeypatch.setattr(sys, "argv", ["bench", "--llm-url", f"http://127.0.0.1:{srv.server_port}",
                                      "--llm-model", "m", "--concurrency", "1", "4", "--n", "40",
                                      "--agents", "20", "--days", "2", "--cache-bust", "--cost-per-hr", "1",
                                      "--out", str(out)])
    try:
        bench.main()
    finally:
        srv.shutdown()
    prompts = [b["messages"][0]["content"] for b in SEEN]
    assert len(prompts) == 1 + 40 + 40 and len(set(prompts)) == len(prompts)
    firsts = [p.split("\n", 1)[0] for p in prompts]
    assert len(set(firsts)) == len(firsts) and all(f.startswith("[request ") for f in firsts)
    r = json.loads(out.read_text())
    assert [x["requests"] for x in r["results"]] == [40, 40] and r["cache_bust"] is True
