"""Ask an LLM for alpha factors, validate them, and add them to the factor library.

    python generate_factors.py --provider gemini --batches 3 --per-batch 20
    python generate_factors.py --provider groq --model qwen/qwen3.8-27b
    python generate_factors.py --provider replay --replay-dir ../llm_runs/seed

Every raw LLM reply is saved under llm_runs/ (prompt + reply + metadata), so any
run can be reproduced exactly with --provider replay.

Library: factors/library.jsonl, one JSON record per proposed factor, including
invalid ones (with the error), because Phase 3 feeds those errors back to the LLM.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from dsl import FactorError, parse
from llm import PROVIDERS, LLMClient, LLMError, extract_json
from prompts import CATEGORIES, PROMPT_VERSION, SYSTEM, user_prompt

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "factors" / "library.jsonl"
RUNS = ROOT / "llm_runs"


def load_library() -> list[dict]:
    if not LIB.exists():
        return []
    return [json.loads(l) for l in LIB.read_text().splitlines() if l.strip()]


def append_library(records: list[dict]) -> None:
    LIB.parent.mkdir(parents=True, exist_ok=True)
    with LIB.open("a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def ingest(reply_text: str, source: dict, library: list[dict]) -> list[dict]:
    """Parse one LLM reply into library records (valid, invalid or duplicate)."""
    seen = {r["expression"] for r in library if r["status"] == "valid"}
    next_id = len(library) + 1
    try:
        obj = extract_json(reply_text)
        items = obj.get("factors", [])
    except (ValueError, json.JSONDecodeError) as e:
        print(f"    could not parse reply as JSON: {e}")
        return []
    if obj.get("_truncated"):
        print(f"    note: reply was cut off; recovered {len(items)} complete factors from it")
    out = []
    for it in items:
        raw = str(it.get("expression", ""))
        rec = {
            "id": f"F{next_id:04d}", "name": str(it.get("name", ""))[:60],
            "category": it.get("category") if it.get("category") in CATEGORIES else "other",
            "raw_expression": raw, "expression": None,
            "hypothesis": str(it.get("hypothesis", ""))[:300],
            **source, "status": None, "error": None,
        }
        try:
            node = parse(raw)
            canon = str(node)
            rec.update(expression=canon, n_nodes=node.size(), depth=node.depth(),
                       lookback=node.max_lookback(), fields=sorted(node.fields()))
            if canon in seen:
                rec["status"] = "duplicate"
            else:
                rec["status"] = "valid"
                seen.add(canon)
        except FactorError as e:
            rec.update(status="invalid", error=str(e))
        out.append(rec)
        next_id += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, choices=[*PROVIDERS, "replay"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--per-batch", type=int, default=20)
    ap.add_argument("--replay-dir", default=str(RUNS / "seed"))
    a = ap.parse_args()

    library = load_library()
    RUNS.mkdir(exist_ok=True)

    if a.provider == "replay":
        files = sorted(Path(a.replay_dir).glob("*.json"))
        if not files:
            raise SystemExit(f"no saved replies in {a.replay_dir}")
        done = {r["run_file"] for r in library}
        for fp in files:
            rel = str(fp.relative_to(ROOT))
            if rel in done:
                print(f"  {rel}: already in library, skipping")
                continue
            run = json.loads(fp.read_text())
            src = {"provider": run["provider"], "model": run["model"], "prompt_version": run["prompt_version"],
                   "run_file": rel, "created_at": run["created_at"]}
            recs = ingest(run["reply"], src, library)
            append_library(recs)
            library += recs
            _report(rel, recs)
        return

    client = LLMClient(a.provider, a.model)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for b in range(1, a.batches + 1):
        existing = [r["expression"] for r in library if r["status"] == "valid"]
        prompt = user_prompt(a.per_batch, b, existing)
        t = time.time()
        print(f"  batch {b}/{a.batches}: asking {client.provider}:{client.model} "
              "(free tiers can take a few minutes) ...", flush=True)
        try:
            resp = client.complete(SYSTEM, prompt)
        except LLMError as e:
            print(f"  batch {b} failed: {e}")
            if b == 1:
                raise SystemExit(1)
            break
        run = {"provider": resp.provider, "model": resp.model, "prompt_version": PROMPT_VERSION,
               "created_at": datetime.now(timezone.utc).isoformat(), "system": SYSTEM, "user": prompt,
               "reply": resp.text, "usage": resp.usage, "seconds": round(time.time() - t, 1)}
        safe_model = resp.model.replace("/", "_").replace(":", "_")
        fp = RUNS / f"{stamp}_{resp.provider}_{safe_model}_b{b:02d}.json"
        fp.write_text(json.dumps(run, indent=2))
        src = {"provider": resp.provider, "model": resp.model, "prompt_version": PROMPT_VERSION,
               "run_file": str(fp.relative_to(ROOT)), "created_at": run["created_at"]}
        recs = ingest(resp.text, src, library)
        append_library(recs)
        library += recs
        _report(fp.name, recs)


def _report(label: str, recs: list[dict]) -> None:
    n = {s: sum(r["status"] == s for r in recs) for s in ("valid", "invalid", "duplicate")}
    print(f"  {label}: {len(recs)} proposed -> {n['valid']} valid, {n['invalid']} invalid, "
          f"{n['duplicate']} duplicate")
    for r in recs:
        if r["status"] == "invalid":
            print(f"      x {r['name']}: {r['error']}")


if __name__ == "__main__":
    main()
