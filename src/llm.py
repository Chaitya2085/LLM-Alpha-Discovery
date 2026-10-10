"""LLM client for many providers, using only the Python standard library.

Free providers (no credit card), all through the same OpenAI-compatible API:
  gemini      Google AI Studio      key: GEMINI_API_KEY      https://aistudio.google.com/apikey
  groq        Groq                  key: GROQ_API_KEY        https://console.groq.com/keys
  cerebras    Cerebras              key: CEREBRAS_API_KEY    https://cloud.cerebras.ai
  openrouter  OpenRouter (":free")  key: OPENROUTER_API_KEY  https://openrouter.ai/keys
  mistral     Mistral               key: MISTRAL_API_KEY     https://console.mistral.ai/api-keys
  ollama      runs on your computer (no key)                 https://ollama.com
Paid:
  anthropic   Claude                key: ANTHROPIC_API_KEY
  openai      OpenAI                key: OPENAI_API_KEY

Keys go in a `.env` file at the project root (git-ignored), e.g. GEMINI_API_KEY=...
Free-tier limits and model names change often. Check what your key can use with:

    python src/llm.py check                     # test every provider that has a key
    python src/llm.py models --provider groq    # list model IDs for one provider
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = "llm-alpha-discovery/1.0"


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    key_env: str | None          # None = no key needed (local)
    default_model: str           # a suggestion; override with --model or <NAME>_MODEL in .env
    free: bool
    signup: str
    style: str = "openai"        # "openai" (chat/completions) or "anthropic" (messages)


PROVIDERS: dict[str, Provider] = {p.name: p for p in [
    Provider("gemini", "https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY",
             "gemini-3.8-flash", True, "https://aistudio.google.com/apikey"),
    Provider("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY",
             "openai/gpt-oss-120b", True, "https://console.groq.com/keys"),
    Provider("cerebras", "https://api.cerebras.ai/v1", "CEREBRAS_API_KEY",
             "qwen-3.8-27b", True, "https://cloud.cerebras.ai"),
    Provider("openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
             "auto-free", True, "https://openrouter.ai/keys"),  # picks a currently free model
    Provider("mistral", "https://api.mistral.ai/v1", "MISTRAL_API_KEY",
             "mistral-small-latest", True, "https://console.mistral.ai/api-keys"),
    Provider("ollama", "http://localhost:11434/v1", None,
             "qwen3:8b", True, "https://ollama.com/download"),
    Provider("anthropic", "https://api.anthropic.com", "ANTHROPIC_API_KEY",
             "claude-sonnet-5-5", False, "https://console.anthropic.com", style="anthropic"),
    Provider("openai", "https://api.openai.com/v1", "OPENAI_API_KEY",
             "", False, "https://platform.openai.com/api-keys"),
]}


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env reader: KEY=value lines; real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.split(" #")[0].strip().strip('"').strip("'")
        if v:
            os.environ.setdefault(k.strip(), v)


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    usage: dict


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, provider: str, model: str | None = None, timeout: int = 900,
                 max_retries: int = 6):
        load_dotenv()
        if provider not in PROVIDERS:
            raise SystemExit(f"unknown provider '{provider}'. Choose from: {', '.join(PROVIDERS)}")
        p = PROVIDERS[provider]
        self.p = p
        self.provider = provider
        self.base_url = os.environ.get(f"{provider.upper()}_BASE_URL", p.base_url).rstrip("/")
        self.model = model or os.environ.get(f"{provider.upper()}_MODEL") or p.default_model
        if not self.model:
            raise SystemExit(f"{provider}: pass --model (or set {provider.upper()}_MODEL in .env).")
        self.timeout = timeout
        self.max_retries = max_retries
        self.key = None
        if p.key_env:
            self.key = os.environ.get(p.key_env)
            if not self.key:
                raise SystemExit(f"{p.key_env} is not set. Get a key at {p.signup} and add it to .env")

    # ------------------------------------------------------------------ http
    def _headers(self) -> dict:
        if self.p.style == "anthropic":
            return {"x-api-key": self.key, "anthropic-version": "2023-06-01"}
        h = {"authorization": f"Bearer {self.key or 'none'}"}
        if self.provider == "openrouter":
            h.update({"x-title": "llm-alpha-discovery"})
        return h

    def _request(self, method: str, url: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        timeouts = 0
        for attempt in range(self.max_retries):
            # Cloudflare (in front of Groq and others) blocks Python's default "Python-urllib"
            # user agent with HTTP 403 "error code: 1010", so name the client explicitly.
            req = urllib.request.Request(url, data=data, method=method,
                                         headers={"content-type": "application/json",
                                                  "user-agent": USER_AGENT, **self._headers()})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors="replace")[:600]
                retryable = e.code in (408, 429, 500, 502, 503, 504, 529)
                if retryable and attempt < self.max_retries - 1:
                    wait = _retry_after(e.headers) or min(90, 5 * 2 ** attempt)
                    print(f"    {self.provider}: HTTP {e.code} (rate limit or busy), waiting {wait:.0f}s",
                          flush=True)
                    time.sleep(wait)
                    continue
                hint = _hint(self.provider, e.code, msg)
                raise LLMError(f"{self.provider} HTTP {e.code}: {msg}{hint}") from None
            except (urllib.error.URLError, TimeoutError) as e:
                is_timeout = isinstance(e, TimeoutError) or "timed out" in str(getattr(e, "reason", ""))
                timeouts += is_timeout
                if attempt < self.max_retries - 1 and timeouts <= 1:  # a slow model gets one more try
                    time.sleep(5)
                    continue
                reason = getattr(e, "reason", e)
                extra = " Is Ollama running? Open the Ollama app first." if self.provider == "ollama" else ""
                raise LLMError(f"could not reach {self.provider} at {url}: {reason}.{extra}") from None
        raise LLMError("unreachable")

    # ------------------------------------------------------------------- api
    def complete(self, system: str, user: str, max_tokens: int = 8000) -> LLMResponse:
        if self.model == "auto-free":
            return self._complete_auto_free(system, user, max_tokens)
        return self._complete(system, user, max_tokens)

    def _complete_auto_free(self, system: str, user: str, max_tokens: int) -> LLMResponse:
        """Try free models in order of preference; skip any that are busy right now.

        Shared free capacity is often rate-limited upstream ("temporarily rate-limited"),
        so a busy model is skipped rather than failing the whole run. The model that
        answers is kept for the rest of the session.
        """
        candidates = self.free_model_candidates()[:6]
        saved_retries, self.max_retries = self.max_retries, 2
        last = None
        try:
            for mid in candidates:
                self.model = mid
                print(f"    {self.provider}: trying free model {mid}", flush=True)
                try:
                    r = self._complete(system, user, max_tokens)
                except LLMError as e:
                    if "429" in str(e) or "rate-limit" in str(e).lower() or "502" in str(e) or "503" in str(e):
                        print(f"    {self.provider}: {mid} is busy, trying the next free model", flush=True)
                        last = e
                        continue
                    raise
                print(f"    {self.provider}: using free model {mid} "
                      f"(choose a fixed one with --model or {self.provider.upper()}_MODEL)", flush=True)
                return r
        finally:
            self.max_retries = saved_retries
        self.model = "auto-free"
        raise LLMError(f"{self.provider}: all {len(candidates)} free models tried are busy right now; "
                       f"try again later. Last error: {str(last)[:200]}")

    def _extra_params(self, max_tokens: int) -> dict:
        """Provider-specific settings that keep long replies from being cut off.

        Groq's default reply limit is only 1024 tokens and gpt-oss spends much of any limit
        on hidden reasoning, so ask for a bigger limit and light reasoning. The limit stays
        under Groq's free 8K tokens-per-minute (prompt ~2-3K + reply 5K).
        """
        m = self.model.lower()
        if self.provider == "groq":
            extra = {"max_completion_tokens": min(max_tokens, 5000)}
            if "gpt-oss" in m:
                extra["reasoning_effort"] = "low"
            return extra
        if self.provider == "cerebras":
            return {"max_completion_tokens": max_tokens}
        return {}

    def _complete(self, system: str, user: str, max_tokens: int) -> LLMResponse:
        if self.p.style == "anthropic":
            r = self._request("POST", f"{self.base_url}/v1/messages",
                              {"model": self.model, "max_tokens": max_tokens, "system": system,
                               "messages": [{"role": "user", "content": user}]})
            text = "".join(b.get("text", "") for b in r.get("content", []) if b.get("type") == "text")
            return LLMResponse(text, self.provider, r.get("model", self.model), r.get("usage", {}))

        body = {"model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        extra = self._extra_params(max_tokens)
        try:
            r = self._request("POST", f"{self.base_url}/chat/completions", {**body, **extra})
        except LLMError as e:
            if not extra or "HTTP 400" not in str(e):
                raise
            print(f"    {self.provider}: model rejected {sorted(extra)}, retrying without them", flush=True)
            r = self._request("POST", f"{self.base_url}/chat/completions", body)
        try:
            choice = r["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"{self.provider}: unexpected reply format: {str(r)[:300]}") from None
        if choice.get("finish_reason") == "length":
            print(f"    {self.provider}: warning, the reply hit the length limit and was cut off; "
                  "complete factors will still be kept", flush=True)
        if isinstance(content, list):  # some providers return content parts
            content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
        return LLMResponse(content or "", self.provider, r.get("model", self.model), r.get("usage") or {})

    def pick_free_model(self) -> str:
        return self.free_model_candidates()[0]

    def free_model_candidates(self) -> list[str]:
        """Currently free text models from the provider's live list (OpenRouter), best first.

        Prefers families not already used by the other default providers, then larger
        models, so the comparison covers different model families.
        """
        r = self._request("GET", f"{self.base_url}/models")
        free = []
        for m in r.get("data", []):
            mid = m.get("id", "")
            pr = m.get("pricing") or {}
            zero = all(str(pr.get(k, "1")).strip() in ("0", "0.0", "0.00") for k in ("prompt", "completion"))
            if not (mid.endswith(":free") or zero):
                continue
            if any(w in mid.lower() for w in ("vision", "embed", "guard", "tts", "audio", "image", "-vl")):
                continue
            free.append(m)
        if not free:
            raise LLMError(f"{self.provider}: no free models are offered right now; pass --model with a paid one")
        families = ["llama", "deepseek", "mistral", "gemma", "nemotron", "glm", "kimi", "qwen", "gpt-oss"]

        def key(m):
            mid = m["id"].lower()
            fam = next((i for i, f in enumerate(families) if f in mid), len(families))
            sizes = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)b\b", mid)]
            return (fam, -(max(sizes) if sizes else 0), -(m.get("context_length") or 0))

        return [m["id"] for m in sorted(free, key=key)]

    def list_models(self) -> list[str]:
        url = f"{self.base_url}/v1/models" if self.p.style == "anthropic" else f"{self.base_url}/models"
        r = self._request("GET", url)
        items = r.get("data", r.get("models", []))
        ids = []
        for m in items:
            mid = m.get("id") or m.get("name") or ""
            ids.append(mid.removeprefix("models/"))
        return sorted(ids)


def _retry_after(headers) -> float | None:
    for h in ("retry-after", "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
        v = headers.get(h) if headers else None
        if not v:
            continue
        try:
            return min(120.0, float(v))
        except ValueError:
            m = re.match(r"(?:(\d+)m)?([\d.]+)s", v)  # e.g. "1m30s", "7.5s"
            if m:
                return min(120.0, 60 * float(m.group(1) or 0) + float(m.group(2)))
    return None


def _hint(provider: str, code: int, msg: str) -> str:
    if code == 403 and "1010" in msg:
        return "\n  -> blocked by Cloudflare's bot filter; update to the latest code (it sends a proper user agent)"
    if code in (401, 403):
        return f"\n  -> check {PROVIDERS[provider].key_env} in .env"
    if "free" in msg.lower() and code in (400, 404):
        return (f"\n  -> that model is no longer free; list free ones: "
                f"python src/llm.py models --provider {provider} --filter :free")
    if code == 404 or "model" in msg.lower() and code == 400:
        return f"\n  -> model not found; run: python src/llm.py models --provider {provider}"
    if code == 413 or "too large" in msg.lower() or "tokens per minute" in msg.lower():
        return "\n  -> request too large for this free tier; try --per-batch 10"
    if code == 402:
        return "\n  -> account balance/credits issue at the provider"
    return ""


# ------------------------------------------------------------ reply parsing
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


def extract_json(text: str, required_key: str = "factors") -> dict:
    """Find the JSON object in a model reply.

    Handles ```json fences, chatter before/after, and <think>...</think> blocks from
    reasoning models (which may themselves contain braces). Prefers an object that
    has `required_key`.
    """
    text = _THINK.sub("", text or "")
    dec = json.JSONDecoder()
    first = None
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            if required_key in obj:
                return obj
            first = first or obj
    # no complete object with the key: the reply may have been cut off mid-list
    salvaged = _salvage_list(text, required_key)
    if salvaged:
        return {required_key: salvaged, "_truncated": True}
    if first is not None:
        return first
    raise ValueError("no JSON object found in reply")


def _salvage_list(text: str, key: str) -> list:
    """Recover the complete items of a JSON list whose reply was cut off mid-way.

    e.g. '{"factors": [{...}, {...}, {"name": "x", "expr' -> the first two items.
    """
    m = re.search(rf'"{re.escape(key)}"\s*:\s*\[', text)
    if not m:
        return []
    dec = json.JSONDecoder()
    items, i = [], m.end()
    while True:
        while i < len(text) and text[i] in " \t\r\n,":
            i += 1
        if i >= len(text) or text[i] != "{":
            break
        try:
            obj, i = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            break  # the cut-off item
        if isinstance(obj, dict):
            items.append(obj)
    return items


# --------------------------------------------------------------------- cli
def _cli() -> None:
    ap = argparse.ArgumentParser(description="Check LLM provider keys and list models.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="send a tiny test request to every provider that has a key")
    mp = sub.add_parser("models", help="list model IDs available to your key")
    mp.add_argument("--provider", required=True, choices=list(PROVIDERS))
    mp.add_argument("--filter", default="", help="only show IDs containing this text, e.g. ':free'")
    a = ap.parse_args()
    load_dotenv()

    if a.cmd == "models":
        ids = LLMClient(a.provider, model="x").list_models()
        ids = [i for i in ids if a.filter in i]
        print("\n".join(ids) if ids else "(no models returned)")
        return

    print(f"{'provider':11s} {'free':5s} {'status':8s} detail")
    for name, p in PROVIDERS.items():
        if p.key_env and not os.environ.get(p.key_env):
            print(f"{name:11s} {str(p.free):5s} {'no key':8s} add {p.key_env} to .env  ({p.signup})")
            continue
        try:
            c = LLMClient(name, max_retries=2, timeout=60)
            t = time.time()
            r = c.complete("Reply with only the JSON object {\"ok\": true}.", "ping", max_tokens=50)
            ok = "ok" in r.text.lower()
            print(f"{name:11s} {str(p.free):5s} {'OK' if ok else 'reply?':8s} model {c.model}, "
                  f"{time.time() - t:.1f}s" + ("" if ok else f", got: {r.text[:60]!r}"))
        except (LLMError, SystemExit) as e:
            if name == "ollama" and "could not reach" in str(e):
                print(f"{name:11s} {str(p.free):5s} {'off':8s} not running (optional: install from {p.signup})")
                continue
            print(f"{name:11s} {str(p.free):5s} {'FAIL':8s} {str(e).splitlines()[0][:150]}")


if __name__ == "__main__":
    _cli()
