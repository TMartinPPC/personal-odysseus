#!/usr/bin/env python3
"""Odysseus harness eval harness (Phase 0 of harness-upgrades plan).

Drives src.agent_loop.stream_agent_loop headless against a configured
OpenAI-compatible endpoint and scores harness-level model quality:

  edit   — file-edit benchmark: first-attempt edit success, rounds-to-success
  read   — read fidelity: correct answers vs confabulation across depths of a
           large file (truncation-hygiene check)
  protocol — tool-call protocol health: malformed-call rate, per-task round
           counts (derived from the other suites' event logs)

Usage (from the repo root, with the venv that has requirements installed):

  python evals/run_evals.py \
    --endpoint http://localhost:8080/v1 \
    --model odysseus-qwen3-4b \
    --api-key-env MY_KEY \
    --suite edit,read \
    --out eval-results/baseline.json

Results are JSON: {config, suites: {edit: {...}, read: {...}}, tasks: [...]}.
Compare two result files to measure a harness change (the whole point: prove
the harness, not the model, caused a quality delta).
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ABSTAIN_MARKERS = ("cannot find", "not present", "doesn't contain", "does not contain",
                   "no mention", "couldn't find", "could not find", "not in the file")


async def _sse_chunks(gen):
    """Consume the agent loop's SSE async generator into (events, final_text)."""
    events = []
    text_parts = []
    async for chunk in gen:
        events.append(chunk)
        if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]"):
            try:
                data = json.loads(chunk[6:])
            except json.JSONDecodeError:
                continue
            if "delta" in data and isinstance(data["delta"], str):
                text_parts.append(data["delta"])
    return events, "".join(text_parts)


def _count_events(events):
    counts = {"tool_start": 0, "tool_output": 0, "malformed": 0, "rounds": 0,
              "loop_breaker": 0, "rounds_exhausted": 0, "errors": 0}
    for chunk in events:
        if not chunk.startswith("data: "):
            continue
        try:
            data = json.loads(chunk[6:])
        except (json.JSONDecodeError, ValueError):
            continue
        t = data.get("type")
        if t == "tool_start":
            counts["tool_start"] += 1
            if data.get("tool") == "__malformed_tool_call__":
                counts["malformed"] += 1
        elif t == "tool_output":
            counts["tool_output"] += 1
        elif t == "agent_step":
            counts["rounds"] += 1
        elif t == "loop_breaker_triggered":
            counts["loop_breaker"] += 1
        elif t == "rounds_exhausted":
            counts["rounds_exhausted"] += 1
        elif t == "error" or (isinstance(data.get("error"), str) and t is None):
            counts["errors"] += 1
    return counts


# ── edit suite ────────────────────────────────────────────────────────────

EDIT_TASKS = [
    {
        "id": "fix-off-by-one",
        "language": "python",
        "seed": (
            "def summarize(numbers):\n"
            "    total = 0\n"
            "    for i in range(len(numbers)):\n"
            "        total += numbers[i]\n"
            "    return total\n"
        ),
        "task": "In numbers.py, the summarize function has an off-by-one bug: it also adds None when the list contains a trailing None sentinel. Change summarize so it skips None values entirely.",
        "check": lambda text: "if numbers[i] is None" in text or "if n is None" in text or "if x is None" in text or "None" in text.split("def summarize")[-1].split("return")[0],
    },
    {
        "id": "rename-variable",
        "language": "python",
        "seed": (
            "def calc_price(qty, unit):\n"
            "    # qty: quantity ordered\n"
            "    # unit: price per unit\n"
            "    return qty * unit\n"
        ),
        "task": "In price.py, rename the parameter 'unit' to 'unit_price' everywhere in calc_price (including the comment and the return line). Keep the docstring comments accurate.",
        "check": lambda text: "unit_price" in text and "def calc_price(qty, unit_price)" in text,
    },
]

async def run_edit_task(endpoint, model, headers, task, max_rounds):
    import tempfile
    ws = tempfile.mkdtemp(prefix="odysseus-eval-")
    filename = "numbers.py" if "numbers" in task["seed"] else "price.py"
    path = os.path.join(ws, filename)
    with open(path, "w") as f:
        f.write(task["seed"])

    import src.agent_loop as al
    messages = [{"role": "user", "content": f"{task['task']} The file is at {path} in the active workspace."}]
    start = time.monotonic()
    try:
        gen = al.stream_agent_loop(
            endpoint, model, messages, headers=headers,
            max_rounds=max_rounds, owner="eval", workspace=ws,
        )
        events, final_text = await _sse_chunks(gen)
    except Exception as e:
        return {"id": task["id"], "ok": False, "error": str(e), "rounds": 0,
                "final_text": "", "events": [], "elapsed_s": round(time.monotonic() - start, 2)}
    elapsed = round(time.monotonic() - start, 2)

    with open(path) as f:
        final_file = f.read()
    ok = False
    try:
        ok = bool(task["check"](final_file))
    except Exception:
        ok = False
    counts = _count_events(events)
    return {
        "id": task["id"], "ok": ok, "rounds": counts["rounds"],
        "tool_starts": counts["tool_start"], "malformed": counts["malformed"],
        "loop_breaker": counts["loop_breaker"], "rounds_exhausted": counts["rounds_exhausted"],
        "elapsed_s": elapsed, "final_text": final_text[:2000], "events": events,
    }

def run_edit_suite(endpoint, model, headers, max_rounds, tasks=None):
    tasks = tasks or EDIT_TASKS
    results = []
    for t in tasks:
        print(f"  [edit] {t['id']} ...", flush=True)
        results.append(asyncio.run(run_edit_task(endpoint, model, headers, t, max_rounds)))
        print(f"    ok={results[-1]['ok']} rounds={results[-1].get('rounds')} elapsed={results[-1].get('elapsed_s')}s")
    n = len(results)
    first_attempt = sum(1 for r in results if r["ok"] and r.get("rounds", 99) <= 2)
    return {
        "n": n,
        "success": sum(1 for r in results if r["ok"]),
        "first_attempt_success": first_attempt,
        "median_rounds": sorted(r.get("rounds", 0) for r in results)[n // 2] if n else 0,
        "malformed_calls": sum(r.get("malformed", 0) for r in results),
        "loop_breakers": sum(r.get("loop_breaker", 0) for r in results),
        "tasks": results,
    }


# ── read-fidelity suite ───────────────────────────────────────────────────

def _build_read_corpus(tmpdir):
    """A ~2600-line file with facts at controlled depths.

    Facts sit at lines ~40, ~700, ~1500, ~2200 and ~2550 so shallow reads
    (old truncation limits) miss the deep ones. The harness records whether
    the model's final answer was correct, an abstention, or a confabulation.
    """
    facts = {
        40:   "FACT-RUBY: The harbor crane was painted ruby red in 1998.",
        700:  "FACT-COBALT: The lighthouse keeper's cat was named Cobalt.",
        1500: "FACT-QUARTZ: The tidal gauge recorded 4,412 millibars on its final day.",
        2200: "FACT-IVORY: The archive key was stored under the ivory floorboard.",
        2550: "FACT-ZEPHYR: The last mail sloop was named Zephyr.",
    }
    lines = []
    fact_lines = sorted(facts)
    for i in range(1, 2601):
        if i in facts:
            lines.append(facts[i])
        else:
            lines.append(f"Log line {i}: routine harbor operations entry, nothing notable happened today.")
    path = os.path.join(tmpdir, "harbor.log")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    questions = [
        (fact_lines[0], "What color was the harbor crane painted, and in what year?", ["ruby red", "1998"]),
        (fact_lines[1], "What was the lighthouse keeper's cat named?", ["Cobalt"]),
        (fact_lines[2], "What pressure did the tidal gauge record on its final day (in millibars)?", ["4412"]),
        (fact_lines[3], "Where was the archive key stored?", ["ivory floorboard"]),
        (fact_lines[4], "What was the last mail sloop named?", ["Zephyr"]),
    ]
    return path, questions

READ_TASKS = 5  # one question per depth

async def run_read_task(endpoint, model, headers, path, question, expect_terms, max_rounds):
    import src.agent_loop as al
    messages = [{"role": "user", "content": (
        f"Read the file {path} in the active workspace and answer this question. "
        f"If the answer is not in the file, say exactly that. Question: {question}"
    )}]
    start = time.monotonic()
    try:
        gen = al.stream_agent_loop(
            endpoint, model, messages, headers=headers,
            max_rounds=max_rounds, owner="eval", workspace=os.path.dirname(path),
        )
        events, final_text = await _sse_chunks(gen)
    except Exception as e:
        return {"ok": False, "verdict": "error", "error": str(e), "answer": "", "elapsed_s": 0}
    elapsed = round(time.monotonic() - start, 2)
    # Classify transport errors before verdict logic — an unreachable endpoint
    # must not count as a confabulation.
    has_error = any("event: error" in c for c in events)
    low = final_text.lower()
    if has_error and not final_text.strip():
        verdict = "error"
    elif not final_text.strip():
        verdict = "empty"
    elif any(term.lower() in low for term in expect_terms if term):
        verdict = "correct"
    elif any(m in low for m in ABSTAIN_MARKERS):
        verdict = "abstained"
    else:
        verdict = "confabulated"
    counts = _count_events(events)
    return {"ok": verdict == "correct", "verdict": verdict, "answer": final_text[:1200],
            "rounds": counts["rounds"], "elapsed_s": elapsed, "events": events}

def run_read_suite(endpoint, model, headers, max_rounds):
    import tempfile
    tmpdir = tempfile.mkdtemp(prefix="odysseus-eval-read-")
    path, questions = _build_read_corpus(tmpdir)
    results = []
    for depth, q, expect_terms in questions:
        print(f"  [read] line~{depth} ...", flush=True)
        r = asyncio.run(run_read_task(endpoint, model, headers, path, q, expect_terms, max_rounds))
        r["depth_line"] = depth
        results.append(r)
        print(f"    verdict={r['verdict']} rounds={r.get('rounds')} elapsed={r.get('elapsed_s')}s")
    n = len(results)
    return {
        "n": n,
        "correct": sum(1 for r in results if r["verdict"] == "correct"),
        "abstained": sum(1 for r in results if r["verdict"] == "abstained"),
        "confabulated": sum(1 for r in results if r["verdict"] == "confabulated"),
        "empty": sum(1 for r in results if r["verdict"] == "empty"),
        "errors": sum(1 for r in results if r["verdict"] in ("error", "empty")),
        "tasks": results,
    }


# ── main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Odysseus harness evals")
    ap.add_argument("--endpoint", required=True, help="OpenAI-compatible base URL, e.g. http://localhost:8080/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key-env", default="", help="Env var name holding the API key (optional)")
    ap.add_argument("--suite", default="edit,read", help="Comma list: edit,read")
    ap.add_argument("--max-rounds", type=int, default=12)
    ap.add_argument("--out", default="", help="Write JSON results here")
    args = ap.parse_args()

    headers = {}
    if args.api_key_env:
        key = os.environ.get(args.api_key_env, "")
        if key:
            headers["Authorization"] = f"Bearer {key}"

    suites = {}
    if "edit" in args.suite:
        print(f"== edit suite ({len(EDIT_TASKS)} tasks) ==", flush=True)
        suites["edit"] = run_edit_suite(args.endpoint, args.model, headers, args.max_rounds)
    if "read" in args.suite:
        print(f"== read suite ({READ_TASKS} questions) ==", flush=True)
        suites["read"] = run_read_suite(args.endpoint, args.model, headers, args.max_rounds)

    result = {
        "schema": "odysseus-harness-evals/1",
        "config": {"endpoint": args.endpoint, "model": args.model, "max_rounds": args.max_rounds},
        "run_at": datetime.now(timezone.utc).isoformat(),
        "suites": suites,
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        # Strip bulky event logs from the summary file but keep them beside it.
        slim = json.loads(json.dumps(result))
        for suite in slim["suites"].values():
            for t in suite.get("tasks", []):
                t.pop("events", None)
        with open(args.out, "w") as f:
            json.dump(slim, f, indent=2)
        raw_path = args.out.replace(".json", "") + "-raw.json"
        with open(raw_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote {args.out} (events in {raw_path})")
    else:
        slim = json.loads(json.dumps(result))
        for suite in slim["suites"].values():
            for t in suite.get("tasks", []):
                t.pop("events", None)
        print(json.dumps(slim, indent=2))

    # Nonzero exit when the suites show degradation signals, so CI can gate.
    failed = False
    if "edit" in suites:
        e = suites["edit"]
        if e["success"] == 0 and e["tasks"] and all(t.get("error") for t in e["tasks"]):
            print("edit suite: every task errored — endpoint/config problem, not a model failure")
            sys.exit(3)
        if e["success"] == 0:
            failed = True
    if "read" in suites:
        r = suites["read"]
        if r["errors"] == r["n"]:
            print("read suite: every question errored — endpoint/config problem, not a model failure")
            sys.exit(3)
        if r["confabulated"] > r["correct"]:
            failed = True
    sys.exit(2 if failed else 0)


if __name__ == "__main__":
    main()
