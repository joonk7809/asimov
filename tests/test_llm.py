"""LLMMind against a local fake OpenAI-compatible server: wire format, schema, replay."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from asiimov.events import read_jsonl, write_jsonl
from asiimov.llm import SCHEMAS, LLMMind
from asiimov.minds import StubMind
from asiimov.replay import replay
from asiimov.sim import TICKS_PER_DAY, Sim

SEEN = []


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append(body)
        kind = body["response_format"]["json_schema"]["name"]
        out = {"plan": {"plan": [{"activity": "work", "start_h": 8, "end_h": 15}]},
               "deliberate": {"action": "go_market", "target": None, "reason": "hungry"},
               "reflect": {"core": [], "summary": "a day"}}[kind]
        data = json.dumps({"model": body["model"], "choices": [{"message": {"content": json.dumps(out)},
                                                                 "finish_reason": "stop"}],
                           "usage": {"prompt_tokens": 900, "completion_tokens": 20}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_llm_mind_drives_world_and_replays_without_server(server, tmp_path):
    SEEN.clear()
    mind = LLMMind(server, "fake/qwen", api_key="k")
    sim = Sim(5, 20, mind)
    sim.run(2, {TICKS_PER_DAY + 60: [{"kind": "fire", "family": 1}]})
    sim.finish()
    kinds = {b["response_format"]["json_schema"]["name"] for b in SEEN}
    assert kinds == {"plan", "deliberate", "reflect"}
    assert all(b["chat_template_kwargs"] == {"enable_thinking": False} for b in SEEN)
    assert mind.stats()["calls"] == len(SEEN)
    mind_events = [e for e in sim.events if e.origin == "mind"]
    assert mind_events and all(e.mind["model"] == "fake/qwen" for e in mind_events)
    path = tmp_path / "log.jsonl"
    write_jsonl(sim.events, str(path))
    n = len(SEEN)
    r = replay(read_jsonl(str(path)))
    assert len(SEEN) == n  # replay never called the server
    assert r.state_hash() == sim.state_hash()


def test_schemas_accept_stub_outputs():
    """The stand-in's outputs fit the LLM schemas, so the two minds speak the same contract."""
    jsonschema = pytest.importorskip("jsonschema")
    from asiimov.bench import collect_prompts
    stub = StubMind()
    for req in collect_prompts(agents=30, days=2, seed=3):
        jsonschema.validate(json.loads(stub.decide(req).raw_output), SCHEMAS[req.kind])


def test_llm_mind_retries_a_200_that_is_not_a_completion(monkeypatch):
    """A proxy page or error JSON with status 200 must be retried, not crash the world mid-run."""
    import asiimov.llm as llm
    replies = iter([b"<html>502</html>", b'{"error": "loading"}', json.dumps(
        {"model": "m", "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}).encode()])

    class Resp:
        def __init__(self, data):
            self.data = data

        def read(self):
            return self.data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(llm.urllib.request, "urlopen", lambda req, timeout: Resp(next(replies)))
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    out = LLMMind("http://x", "m", retries=3).post("/v1/chat/completions", {})
    assert out["choices"][0]["message"]["content"] == "{}"


class SlowFake(Fake):
    """Answers after a random delay, so concurrent replies arrive out of order. Tracks requests in flight."""
    lock = threading.Lock()
    inflight = 0
    peak = 0

    def do_POST(self):
        import random
        import time
        with SlowFake.lock:
            SlowFake.inflight += 1
            SlowFake.peak = max(SlowFake.peak, SlowFake.inflight)
        time.sleep(random.uniform(0, 0.003))
        try:
            super().do_POST()
        finally:
            with SlowFake.lock:
                SlowFake.inflight -= 1


class Server(ThreadingHTTPServer):
    request_queue_size = 128  # the default backlog of 5 drops bursts of connects, which then retry after 1 s


def test_concurrent_calls_to_a_slow_server_give_the_serial_log(tmp_path):
    srv = Server(("127.0.0.1", 0), SlowFake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}"
    try:
        fire = {TICKS_PER_DAY + 60: [{"kind": "fire", "family": 1}]}
        SlowFake.peak = 0
        fast = Sim(5, 30, LLMMind(url, "fake/qwen"), workers=16)
        fast.run(2, fire)
        assert SlowFake.peak > 1, "calls were not concurrent"
        serial = Sim(5, 30, LLMMind(url, "fake/qwen"), workers=1)
        serial.run(2, fire)
    finally:
        srv.shutdown()
    assert [e.to_json() for e in fast.events] == [e.to_json() for e in serial.events]
    fast.finish()
    path = tmp_path / "log.jsonl"
    write_jsonl(fast.events, str(path))
    assert replay(read_jsonl(str(path))).state_hash() == fast.state_hash()
