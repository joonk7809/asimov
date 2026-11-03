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

ACTIVITIES = ["work", "socialize", "home", "rest", "explore"]
ACTIONS = ["buy_food", "go_market", "ask_kin", "steal", "wait"]

SCHEMAS = {
    "plan": {
        "type": "object",
        "properties": {"plan": {"type": "array", "maxItems": 4, "items": {
            "type": "object",
            "properties": {"activity": {"type": "string", "enum": ACTIVITIES},
                           "start_h": {"type": "integer", "minimum": 0, "maximum": 23},
                           "end_h": {"type": "integer", "minimum": 1, "maximum": 24}},
            "required": ["activity", "start_h", "end_h"], "additionalProperties": False}}},
        "required": ["plan"], "additionalProperties": False,
    },
    "deliberate": {
        "type": "object",
        "properties": {"action": {"type": "string", "enum": ACTIONS},
                       "target": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                       "reason": {"type": "string", "maxLength": 160}},
        "required": ["action", "target", "reason"], "additionalProperties": False,
    },
    "reflect": {
        "type": "object",
        "properties": {"core": {"type": "array", "maxItems": 2, "items": {"type": "string", "maxLength": 160}},
                       "summary": {"type": "string", "maxLength": 240}},
        "required": ["core", "summary"], "additionalProperties": False,
    },
}


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
            "seed": int(req.prompt_hash[:8], 16),
            # Qwen3 hybrid models: no <think> block; deliberation is the answer itself.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if self.structured:
            b["response_format"] = {"type": "json_schema", "json_schema": {
                "name": req.kind, "schema": SCHEMAS[req.kind], "strict": True}}
        return b

    def post(self, path: str, body: dict) -> dict:
        data = json.dumps(body).encode()
        headers = {"Content-Type": "application/json"}
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
        return MindReply(text.strip(), resp.get("model", self.model), sampling)

    def stats(self) -> dict:
        if not self.calls:
            return {"calls": 0}
        lat = sorted(c["latency_s"] for c in self.calls)
        pt = sum(c["prompt_tokens"] or 0 for c in self.calls)
        ct = sum(c["completion_tokens"] or 0 for c in self.calls)
        return {"calls": len(lat), "latency_p50_s": lat[len(lat) // 2], "latency_p95_s": lat[int(len(lat) * .95)],
                "wall_s_in_model": round(sum(lat), 1), "prompt_tokens": pt, "completion_tokens": ct,
                "truncated": sum(1 for c in self.calls if c["finish_reason"] == "length")}
