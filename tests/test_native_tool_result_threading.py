"""Native tool-call results must be threaded by CONVERTED-call position.

When an OpenAI/Anthropic model emits several tool_calls in one round and one
fails to convert (hallucinated name or bad-JSON args), harness-upgrades
surfaces it as a __malformed_tool_call__ pseudo block that executes as an
actionable error result — the model gets feedback instead of a silently
dropped call. Alignment stays 1:1: every native call yields exactly one
block and one converted entry, so _append_tool_results indexes results by
native-call position correctly.
"""
import src.agent_loop as al


def test_resolve_returns_converted_calls_aligned():
    native = [
        {"name": "bogus_unknown_tool", "arguments": "{}", "id": "A"},
        {"name": "web_search", "arguments": '{"query": "hello"}', "id": "B"},
    ]
    tool_blocks, used_native, converted = al._resolve_tool_blocks("", native, 1)
    assert used_native is True
    assert len(tool_blocks) == 2
    assert [c["name"] for c in converted] == ["bogus_unknown_tool", "web_search"]
    assert len(converted) == len(tool_blocks)  # aligned 1:1
    # The failed call became a malformed pseudo block, not a silent drop.
    assert tool_blocks[0].tool_type == "__malformed_tool_call__"
    assert tool_blocks[1].tool_type == "web_search"


def test_append_threads_result_to_correct_tool_call_id():
    messages = []
    converted = [{"id": "B", "name": "web_search", "arguments": "{}"}]
    al._append_tool_results(
        messages, "some response", converted,
        ["RESULT"], ["RESULT"], True, 1,
    )
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "B"
    assert tool_msgs[0]["content"] == "RESULT"
    asst = next(m for m in messages if m.get("role") == "assistant")
    assert [tc["id"] for tc in asst["tool_calls"]] == ["B"]
