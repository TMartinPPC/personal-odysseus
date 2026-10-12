"""Steering queue + loop delivery tests (harness-upgrades Phase 4).

Covers:
1. src.steering module semantics (enqueue/drain/overflow/clear).
2. stream_agent_loop drains queued steering messages at the round boundary
   and appends them as ordinary user turns after the previous round's tool
   results — pi/opencode steering semantics.
3. Malformed native tool calls surface as actionable error results instead
   of being silently dropped (oh-my-pi "errors that teach" pattern).
"""
import asyncio
import json
import sys
from unittest.mock import MagicMock

# Mock heavy deps so importing src.agent_loop doesn't load the full app stack
# (same pattern as test_loop_breaker_runaway.py).
_MOCKED = [
    'sqlalchemy', 'sqlalchemy.orm', 'sqlalchemy.ext', 'sqlalchemy.ext.declarative',
    'sqlalchemy.ext.hybrid', 'sqlalchemy.sql', 'sqlalchemy.sql.expression',
    'src.database', 'core.models', 'core.database',
]
for _m in _MOCKED:
    sys.modules.setdefault(_m, MagicMock())

from src import steering
import src.agent_loop as al


# ── steering module ──────────────────────────────────────────────────────

def test_enqueue_drain_roundtrip():
    steering.clear("s1")
    assert steering.enqueue("s1", "focus on tests next") is True
    assert steering.pending("s1") == 1
    assert steering.drain("s1") == ["focus on tests next"]
    assert steering.pending("s1") == 0
    assert steering.drain("s1") == []


def test_enqueue_rejects_empty():
    assert steering.enqueue("s2", "   ") is False
    assert steering.enqueue("", "hello") is False


def test_queue_overflow_drops_oldest():
    steering.clear("s3")
    for i in range(steering.MAX_PENDING_STEERS + 2):
        assert steering.enqueue("s3", f"msg {i}") is True
    drained = steering.drain("s3")
    assert len(drained) == steering.MAX_PENDING_STEERS
    # Oldest two were dropped on overflow
    assert drained[0] == "msg 2"
    assert drained[-1] == "msg 9"


# ── loop delivery ────────────────────────────────────────────────────────

def _sse(obj):
    return "data: " + json.dumps(obj) + "\n\n"


def _run_agent(monkeypatch, *, rounds, fake_streams, steers_by_round):
    """Drive stream_agent_loop with a scripted fake LLM stream.

    fake_streams: list of chunk-lists, one per LLM call.
    steers_by_round: dict round -> list of steering messages the getter returns.
    Returns (chunks, captured_message_lists).
    """
    monkeypatch.setattr(al, "get_setting", lambda key, default=None: default, raising=False)
    monkeypatch.setattr(al, "get_mcp_manager", lambda: None, raising=False)
    monkeypatch.setattr(al, "estimate_tokens", lambda *a, **k: 10, raising=False)
    monkeypatch.setattr(al, "blocked_tools_for_owner", lambda owner: set(), raising=False)

    captured = []
    call_idx = {"n": 0}

    async def _fake_stream(_candidates, messages, **kwargs):
        idx = call_idx["n"]
        call_idx["n"] += 1
        captured.append([dict(m) for m in messages])
        for chunk in fake_streams[min(idx, len(fake_streams) - 1)]:
            yield chunk

    monkeypatch.setattr(al, "stream_llm_with_fallback", _fake_stream, raising=False)

    async def _fake_execute(block, **kwargs):
        return (f"{block.tool_type}: executed", {"output": "ok", "exit_code": 0})

    monkeypatch.setattr(al, "execute_tool_block", _fake_execute, raising=False)

    def _getter():
        return steers_by_round.get(call_idx["n"] + 1, [])

    async def _run():
        gen = al.stream_agent_loop(
            "https://api.openai.com/v1", "gpt-test",
            [{"role": "user", "content": "do the thing"}],
            max_rounds=rounds, relevant_tools={"get_workspace"}, owner="admin",
            steering_getter=_getter,
        )
        return [c async for c in gen], captured

    return asyncio.run(_run())


def test_steering_message_delivered_after_tool_results(monkeypatch):
    round1 = [
        _sse({"type": "tool_calls", "calls": [{"id": "c1", "name": "get_workspace", "arguments": "{}"}]}),
        "data: [DONE]\n\n",
    ]
    round2 = [_sse({"delta": "done"}), "data: [DONE]\n\n"]
    steers = {2: ["stop — check the tests instead"]}
    chunks, captured = _run_agent(
        monkeypatch, rounds=3,
        fake_streams=[round1, round2],
        steers_by_round=steers,
    )
    # Round 2's request must contain the steer as a user message appended
    # after the assistant tool-call turn and its tool result.
    r2 = captured[1]
    steer_idx = next(
        i for i, m in enumerate(r2)
        if m["role"] == "user" and m.get("content") == "stop — check the tests instead"
    )
    # It must come after the assistant tool_calls message and its tool result.
    tool_result_idx = next(i for i, m in enumerate(r2) if m.get("role") == "tool")
    assert steer_idx > tool_result_idx
    assert any('"delta": "done"' in c or '"delta":"done"' in c for c in chunks)


def test_steering_getter_failure_never_breaks_run(monkeypatch):
    round1 = [_sse({"delta": "hi"}), "data: [DONE]\n\n"]

    def _boom():
        raise RuntimeError("queue exploded")

    chunks, _ = _run_agent(monkeypatch, rounds=1, fake_streams=[round1], steers_by_round={})
    # Rerun with a raising getter to prove isolation.
    monkeypatch.setattr(al, "get_setting", lambda key, default=None: default, raising=False)
    monkeypatch.setattr(al, "get_mcp_manager", lambda: None, raising=False)
    monkeypatch.setattr(al, "estimate_tokens", lambda *a, **k: 10, raising=False)
    monkeypatch.setattr(al, "blocked_tools_for_owner", lambda owner: set(), raising=False)

    async def _fake_stream(_candidates, messages, **kwargs):
        for chunk in round1:
            yield chunk

    monkeypatch.setattr(al, "stream_llm_with_fallback", _fake_stream, raising=False)

    async def _run():
        gen = al.stream_agent_loop(
            "https://api.openai.com/v1", "gpt-test",
            [{"role": "user", "content": "hello"}],
            max_rounds=1, relevant_tools=None, owner="admin",
            steering_getter=_boom,
        )
        return [c async for c in gen]

    out = asyncio.run(_run())
    assert any("hi" in c for c in out)


# ── malformed native call feedback ───────────────────────────────────────

def test_malformed_native_call_becomes_error_result(monkeypatch):
    from src.agent_loop import _resolve_tool_blocks

    blocks, used_native, converted = _resolve_tool_blocks(
        "", [{"name": "edit_file", "arguments": '{"path": "a.txt", "old_string": '}],
        1, is_api_model=True,
    )
    assert used_native is True
    assert len(blocks) == 1 and len(converted) == 1
    assert blocks[0].tool_type == "__malformed_tool_call__"

    from src.tool_execution import execute_tool_block, NO_TOOL_SECURITY_CONTEXT

    async def _run():
        return await execute_tool_block(blocks[0], security_context=NO_TOOL_SECURITY_CONTEXT)

    desc, result = asyncio.run(_run())
    assert result["exit_code"] == 1
    assert "NOT executed" in result["error"]
    assert "edit_file" in result["error"]
    assert "valid JSON" in result["error"]


def test_unknown_native_tool_name_gets_did_you_mean(monkeypatch):
    from src.agent_loop import _resolve_tool_blocks

    blocks, _, _ = _resolve_tool_blocks(
        "", [{"name": "list_sesssions", "arguments": "{}"}],
        1, is_api_model=True,
    )
    assert blocks[0].tool_type == "__malformed_tool_call__"
    info = json.loads(blocks[0].content)
    assert info["reason"] == "unknown tool name"

    from src.tool_execution import execute_tool_block, NO_TOOL_SECURITY_CONTEXT

    async def _run():
        return await execute_tool_block(blocks[0], security_context=NO_TOOL_SECURITY_CONTEXT)

    desc, result = asyncio.run(_run())
    assert "Did you mean" in result["error"] or "did you mean" in result["error"].lower()


def test_empty_round_retry_directive_injected(monkeypatch):
    # Round 1: completely empty (no text, no reasoning, no tool calls).
    # Round 2: after the retry directive, the model answers.
    empty_round = ["data: [DONE]\n\n"]
    answer_round = [_sse({"delta": "here is the answer"}), "data: [DONE]\n\n"]
    chunks, captured = _run_agent(
        monkeypatch, rounds=3,
        fake_streams=[empty_round, answer_round],
        steers_by_round={},
    )
    # Round 2's request must contain the retry directive as a system message.
    r2 = captured[1]
    assert any(
        m["role"] == "system" and "previous response was completely empty" in (m.get("content") or "")
        for m in r2
    )
    assert any('"delta": "here is the answer"' in c for c in chunks)


def test_empty_round_retry_capped(monkeypatch):
    # Four consecutive empty rounds: directive injected after rounds 1-3,
    # then the streak stops nudging (round 5 request has no new directive
    # beyond the third) and the run ends via the empty-response fallback.
    empty_round = ["data: [DONE]\n\n"]
    chunks, captured = _run_agent(
        monkeypatch, rounds=5,
        fake_streams=[empty_round],
        steers_by_round={},
    )
    directive_rounds = []
    for i, msgs in enumerate(captured, start=1):
        n_directives = sum(
            1 for m in msgs
            if m.get("role") == "system" and "previous response was completely empty" in (m.get("content") or "")
        )
        if n_directives:
            directive_rounds.append((i, n_directives))
    # Directives appear in the requests for rounds 2, 3, 4 — then capped.
    assert [r for r, _ in directive_rounds] == [2, 3, 4]
    assert any("empty response" in c for c in chunks)
