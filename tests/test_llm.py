"""Tests for the LLM client, using a local stand-in server (no API keys, no internet).

Run:  python tests/test_llm.py   (or pytest)
"""
import http.server
import json
import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llm import PROVIDERS, LLMClient, LLMError, extract_json  # noqa: E402

REPLY = ('<think>maybe {"factors": "draft"} ... let me check {braces}</think>\n'
         'Here you go:\n```json\n{"factors": [{"name": "a", "category": "momentum", '
         '"expression": "Pct(close, 20)", "hypothesis": "h"}]}\n```')
STATE = {"calls": [], "fail_next": 0, "busy_models": set()}


class Handler(http.server.BaseHTTPRequestHandler):
    def _send(self, code, obj, headers=None):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _blocked_agent(self):
        # mimic Cloudflare's bot filter in front of Groq: Python's default agent is refused
        if "python-urllib" in (self.headers.get("user-agent") or "").lower():
            self._send(403, {"error": "error code: 1010"})
            return True
        return False

    def do_GET(self):
        STATE["calls"].append(("GET", self.path, dict(self.headers), None))
        if self._blocked_agent():
            return
        self._send(200, {"data": [
            {"id": "models/model-b"}, {"id": "model-a"},
            {"id": "acme/paid-405b", "pricing": {"prompt": "0.000002", "completion": "0.000004"}},
            {"id": "qwen/qwen3-235b:free", "pricing": {"prompt": "0", "completion": "0"}},
            {"id": "meta-llama/llama-4-8b:free", "pricing": {"prompt": "0", "completion": "0"}},
            {"id": "meta-llama/llama-4-70b:free", "pricing": {"prompt": "0", "completion": "0"}},
            {"id": "meta-llama/llama-4-90b-vision:free", "pricing": {"prompt": "0", "completion": "0"}},
        ]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        STATE["calls"].append(("POST", self.path, dict(self.headers), body))
        if self._blocked_agent():
            return
        if body.get("model") in STATE["busy_models"]:
            return self._send(429, {"error": {"message": "Provider returned error", "code": 429,
                                              "metadata": {"raw": f"{body['model']} is temporarily rate-limited upstream"}}})
        if STATE["fail_next"]:
            STATE["fail_next"] -= 1
            return self._send(429, {"error": "slow down"}, {"retry-after": "1"})
        if STATE.get("reject_extras") and ("max_completion_tokens" in body or "reasoning_effort" in body):
            return self._send(400, {"error": "unsupported parameter"})
        if self.headers.get("authorization") == "Bearer bad":
            return self._send(401, {"error": "invalid key"})
        if self.path.endswith("/v1/messages"):
            return self._send(200, {"model": body["model"], "content": [{"type": "text", "text": REPLY}],
                                    "usage": {"input_tokens": 3}})
        return self._send(200, {"model": body["model"], "choices": [{"message": {"content": REPLY}}],
                                "usage": {"prompt_tokens": 3}})

    def log_message(self, *a):
        pass


def _server():
    srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_port}"


URL = _server()


def _env(provider, key="test-key", path=""):
    p = PROVIDERS[provider]
    if p.key_env:
        os.environ[p.key_env] = key
    os.environ[f"{provider.upper()}_BASE_URL"] = URL + path


def test_extract_json_handles_think_fences_and_chatter():
    obj = extract_json(REPLY)
    assert obj["factors"][0]["expression"] == "Pct(close, 20)"
    assert extract_json('{"x": 1} then {"factors": []}')["factors"] == []
    try:
        extract_json("no json here")
        raise AssertionError("should fail")
    except ValueError:
        pass


def test_salvages_complete_factors_from_cut_off_reply():
    cut = ('{\n "factors": [\n {"name": "a", "category": "momentum", "expression": "Pct(close, 20)", '
           '"hypothesis": "h"},\n {"name": "b", "category": "volume", "expression": "-Std(Log(volume), 40)", '
           '"hypothesis": "h {with braces}"},\n {"name": "c", "category": "rev')
    obj = extract_json(cut)
    assert obj.get("_truncated") and [f["name"] for f in obj["factors"]] == ["a", "b"]


def test_groq_asks_for_long_replies_and_light_reasoning():
    _env("groq")
    STATE["calls"].clear()
    LLMClient("groq", model="openai/gpt-oss-120b").complete("s", "u")
    body = STATE["calls"][-1][3]
    assert body["max_completion_tokens"] == 5000 and body["reasoning_effort"] == "low"
    LLMClient("groq", model="qwen/qwen3.8-27b").complete("s", "u")
    body = STATE["calls"][-1][3]
    assert "reasoning_effort" not in body and body["max_completion_tokens"] == 5000


def test_falls_back_when_provider_rejects_extra_settings():
    _env("groq")
    STATE["reject_extras"] = True
    try:
        r = LLMClient("groq", model="openai/gpt-oss-120b").complete("s", "u")
        assert "factors" in r.text and "max_completion_tokens" not in STATE["calls"][-1][3]
    finally:
        STATE["reject_extras"] = False


def test_every_openai_style_provider_roundtrip():
    for name, p in PROVIDERS.items():
        if p.style != "openai":
            continue
        _env(name)
        STATE["calls"].clear()
        c = LLMClient(name, model="m1")
        r = c.complete("sys", "user")
        assert extract_json(r.text)["factors"], name
        method, path, headers, body = STATE["calls"][-1]
        h = {k.lower(): v for k, v in headers.items()}
        assert path == "/chat/completions", (name, path)
        assert body["messages"][0] == {"role": "system", "content": "sys"}
        assert body["model"] == "m1"
        assert h["authorization"].startswith("Bearer ")


def test_anthropic_style():
    _env("anthropic")
    STATE["calls"].clear()
    r = LLMClient("anthropic", model="claude-x").complete("sys", "user", max_tokens=123)
    assert extract_json(r.text)["factors"]
    _, path, headers, body = STATE["calls"][-1]
    h = {k.lower(): v for k, v in headers.items()}
    assert path == "/v1/messages" and h["x-api-key"] == "test-key"
    assert h["anthropic-version"] == "2023-06-01"
    assert body["system"] == "sys" and body["max_tokens"] == 123


def test_retries_on_rate_limit_then_succeeds():
    _env("groq")
    STATE["fail_next"] = 2
    r = LLMClient("groq", model="m").complete("s", "u")
    assert "factors" in r.text and STATE["fail_next"] == 0


def test_bad_key_gives_clear_error():
    _env("gemini", key="bad")
    try:
        LLMClient("gemini", model="m").complete("s", "u")
        raise AssertionError("should fail")
    except LLMError as e:
        assert "401" in str(e) and "GEMINI_API_KEY" in str(e)
    _env("gemini")


def test_list_models_strips_prefix_and_sorts():
    _env("gemini")
    ids = LLMClient("gemini", model="m").list_models()
    assert ids[:2] == ["acme/paid-405b", "meta-llama/llama-4-70b:free"] and "model-a" in ids


def test_sends_named_user_agent_not_python_default():
    _env("groq")
    STATE["calls"].clear()
    LLMClient("groq", model="m").complete("s", "u")   # the mock refuses Python-urllib with 1010
    ua = {k.lower(): v for k, v in STATE["calls"][-1][2].items()}["user-agent"]
    assert "python-urllib" not in ua.lower()


def test_openrouter_auto_picks_a_free_text_model():
    _env("openrouter")
    c = LLMClient("openrouter")                 # default model is "auto-free"
    assert c.model == "auto-free"
    r = c.complete("s", "u")
    # free only, no vision models, preferred family (llama) first, larger size wins
    assert c.model == "meta-llama/llama-4-70b:free", c.model
    assert STATE["calls"][-1][3]["model"] == c.model and "factors" in r.text


def test_openrouter_skips_busy_free_models():
    _env("openrouter")
    STATE["busy_models"] = {"meta-llama/llama-4-70b:free", "meta-llama/llama-4-8b:free"}
    try:
        c = LLMClient("openrouter")
        r = c.complete("s", "u")
        assert c.model == "qwen/qwen3-235b:free", c.model   # next free model after the busy ones
        assert "factors" in r.text and c.max_retries == 6       # retries restored afterwards
        STATE["busy_models"] = {"meta-llama/llama-4-70b:free", "meta-llama/llama-4-8b:free",
                                "qwen/qwen3-235b:free"}
        c2 = LLMClient("openrouter")
        try:
            c2.complete("s", "u")
            raise AssertionError("should fail when every free model is busy")
        except LLMError as e:
            assert "busy" in str(e)
    finally:
        STATE["busy_models"] = set()


def test_missing_key_exits_with_signup_link():
    os.environ["CEREBRAS_API_KEY"] = ""  # empty (not unset) so a real .env key cannot leak in
    try:
        LLMClient("cerebras")
        raise AssertionError("should exit")
    except SystemExit as e:
        assert "CEREBRAS_API_KEY" in str(e) and "cerebras.ai" in str(e)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\nall {len(tests)} tests passed")
