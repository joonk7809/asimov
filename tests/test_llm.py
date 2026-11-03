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
