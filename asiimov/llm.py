"""LLM mind: an open model behind an OpenAI-compatible server (vLLM).

The world treats this exactly like any other mind: it sends the assembled prompt and
records the raw output verbatim, so replay never needs the model again. Output shape is
enforced server-side with a JSON schema per request kind (vLLM structured outputs); the
world still validates every field and issues receipts, because a schema can't check facts.
"""
import http.client
import json
import time
import urllib.error
import urllib.request

from .minds import MindReply, MindRequest
from .sim import ACTIVITIES as _ACTIVITIES, STEP_DOS

ACTIVITIES = list(_ACTIVITIES)

NO_TARGET = ["buy_food", "go_market", "work", "socialize", "home", "rest", "explore", "wait"]
# Person actions and the ids each may name: family here for ask_kin (the map's place check), people here
# for ask_food, buy_from and steal, and anyone the prompt lists (family, here, People I know) for the rest.
# A deliberation's view "offer" (engine 10) cuts both lists to what can work now.
PERSON_ACTIONS = (("ask_kin", "kin_here"), ("ask_food", "present"), ("buy_from", "present"), ("steal", "present"),
                  ("give_food", "people"), ("give_coin", "people"), ("seek", "people"), ("avoid", "people"))

# No maxLength on strings: vLLM 0.11 structured outputs (xgrammar) decode ~15x slower with it
# (measured on Qwen3-4B, A6000: reflect 18.5 s vs 1.7 s). max_tokens bounds the reply and the
# world truncates every stored string, so the cap adds nothing.
SCHEMAS = {
    "plan": {
        "type": "object",
        # minItems: a whole waking day, not one short block (one-block plans were 700/700 at eccd95d).
        "properties": {"plan": {"type": "array", "minItems": 2, "maxItems": 6, "items": {
            "type": "object",
            "properties": {"activity": {"type": "string", "enum": ACTIVITIES},
                           "start_h": {"type": "integer", "minimum": 0, "maximum": 23},
                           "end_h": {"type": "integer", "minimum": 1, "maximum": 24}},
            "required": ["activity", "start_h", "end_h"], "additionalProperties": False}}},
        "required": ["plan"], "additionalProperties": False,
    },
    "reflect": {
        "type": "object",
        "properties": {"core": {"type": "array", "maxItems": 2, "items": {"type": "string"}},
                       "summary": {"type": "string"}},
        "required": ["core", "summary"], "additionalProperties": False,
    },
}


def _deliberate_branch(actions: list[str], target: dict, then: dict, goal: dict | None = None) -> dict:
    return {"type": "object",
            "properties": {"action": {"type": "string", "enum": actions}, "target": target,
                           "reason": {"type": "string"}, "goal": goal or {"type": "string"}, "then": then},
            "required": ["action", "target", "reason", "goal", "then"], "additionalProperties": False}


def _ids(view: dict, key: str) -> list[int]:
    if key == "kin_here":
        return sorted(x["id"] for x in view["present"] if x["kin"])
    return sorted(view.get(key, [])) if key == "people" else sorted(x["id"] for x in view[key])


def schema_for(req: MindRequest) -> dict:
    """The reply schema for one request. A deliberation's target must be an id its prompt lists:
    family here for ask_kin, people here for steal, anyone listed for the other person actions. An action
    with nobody to aim at is left out. Intent steps may name anyone listed, or nobody; the world drops
    a person step without a target."""
    if req.kind != "deliberate":
        return SCHEMAS[req.kind]
    people = _ids(req.view, "people")
    offer = req.view.get("offer", NO_TARGET + [x for x, _ in PERSON_ACTIONS])
    if (req.view.get("occasion") or {}).get("why") == "work_end":  # D2: a one-off, so no goal and no steps
        return {"anyOf": [_deliberate_branch([x for x in NO_TARGET if x in offer], {"type": "null"},
                                             {"type": "array", "maxItems": 0}, {"type": "string", "enum": [""]})]}
    # Steps follow the offer too: a shop step buys like go_market, and buy_from needs the same coin.
    dos = [d for d in STEP_DOS if not (d == "shop" and "go_market" not in offer
                                       or d == "buy_from" and "buy_from" not in offer)]
    step_target = {"anyOf": [{"type": "null"}, {"type": "integer", "enum": people}]} if people else {"type": "null"}
    then = {"type": "array", "maxItems": 3, "items": {
        "type": "object", "properties": {"do": {"type": "string", "enum": dos}, "target": step_target},
        "required": ["do", "target"], "additionalProperties": False}}
    branches = [_deliberate_branch([x for x in NO_TARGET if x in offer], {"type": "null"}, then)]
    for action, key in PERSON_ACTIONS:
        ids = _ids(req.view, key)
        if ids and action in offer:
            branches.append(_deliberate_branch([action], {"type": "integer", "enum": ids}, then))
    return {"anyOf": branches}


class LLMMind:
    """Calls /v1/chat/completions. Thread-safe; keeps simple per-call stats for cost reports."""

    def __init__(self, base_url: str, model: str, temperature: float = 0.7, top_p: float = 0.9,
                 max_tokens: int = 200, timeout: float = 120.0, retries: int = 3, structured: bool = True,
                 api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature, self.top_p, self.max_tokens = temperature, top_p, max_tokens
        self.timeout, self.retries, self.structured = timeout, retries, structured
        self.api_key = api_key
        self.calls: list[dict] = []  # {kind, latency_s, prompt_tokens, completion_tokens}

    def body(self, req: MindRequest) -> dict:
        b = {
            "model": self.model,
            "messages": [{"role": "user", "content": req.prompt}],
            "temperature": self.temperature, "top_p": self.top_p, "max_tokens": self.max_tokens,
            # Per-request seed from the prompt. vLLM does not promise identical samples across different
            # batch compositions, so this is not a reproducibility guarantee; replay uses the recording.
            # asiimov.probe sets req.seed to re-ask one prompt with independent samples.
            "seed": req.seed if req.seed is not None else int(req.prompt_hash[:8], 16),
            # Qwen3 hybrid models: no <think> block; deliberation is the answer itself.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if req.frequency_penalty_milli:  # only the T21 reflect retry; every first ask is unchanged
            b["frequency_penalty"] = req.frequency_penalty_milli / 1000
        if self.structured:
            b["response_format"] = {"type": "json_schema", "json_schema": {
                "name": req.kind, "schema": schema_for(req), "strict": True}}
        return b

    def post(self, path: str, body: dict) -> dict:
        data = json.dumps(body).encode()
        headers = {"Content-Type": "application/json", "User-Agent": "asiimov-mind/0.1"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last = None
        for attempt in range(self.retries + 1):
            try:
                r = urllib.request.Request(self.base_url + path, data=data, headers=headers)
                with urllib.request.urlopen(r, timeout=self.timeout) as resp:
                    out = json.loads(resp.read())
                if not isinstance(out, dict) or not out.get("choices"):
                    raise ValueError(f"no choices in server reply: {str(out)[:200]}")
                return out
            # HTTPException: truncated body (IncompleteRead). ValueError: a 200 whose body isn't a completion.
            except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException, ValueError) as e:
                if isinstance(e, urllib.error.HTTPError) and 400 <= e.code < 500 and e.code != 429:
                    raise RuntimeError(f"LLM server rejected request: {e.code} {e.read()[:500]!r}") from e
                last = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM server unreachable after {self.retries + 1} tries: {last}")

    def decide(self, req: MindRequest) -> MindReply:
        body = self.body(req)
        t0 = time.monotonic()
        resp = self.post("/v1/chat/completions", body)
        dt = time.monotonic() - t0
        choice = resp["choices"][0]
        text = choice["message"].get("content") or ""
        usage = resp.get("usage") or {}
        self.calls.append({"kind": req.kind, "latency_s": round(dt, 4),
                           "prompt_tokens": usage.get("prompt_tokens"),
                           "completion_tokens": usage.get("completion_tokens"),
                           "finish_reason": choice.get("finish_reason")})
        sampling = {"temperature": self.temperature, "top_p": self.top_p, "max_tokens": self.max_tokens,
                    "seed": body["seed"], "structured": self.structured}
        if "frequency_penalty" in body:
            sampling["frequency_penalty"] = body["frequency_penalty"]
        return MindReply(text.strip(), resp.get("model", self.model), sampling, choice.get("finish_reason"))

    def stats(self) -> dict:
        if not self.calls:
            return {"calls": 0}
        lat = sorted(c["latency_s"] for c in self.calls)
        pt = sum(c["prompt_tokens"] or 0 for c in self.calls)
        ct = sum(c["completion_tokens"] or 0 for c in self.calls)
        return {"calls": len(lat), "latency_p50_s": lat[len(lat) // 2], "latency_p95_s": lat[int(len(lat) * .95)],
                "wall_s_in_model": round(sum(lat), 1), "prompt_tokens": pt, "completion_tokens": ct,
                "truncated": sum(1 for c in self.calls if c["finish_reason"] == "length")}
