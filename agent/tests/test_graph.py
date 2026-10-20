"""Draft-graph tests with a scripted LLM (no network, no spend)."""

import json

import pytest

pytest.importorskip("langgraph")

from agent.graph import GraphConfig, build_decision_graph  # noqa: E402
from agent.llm import LLMReply, ScriptedLLM, ToolCall  # noqa: E402
from agent.schema import SchemaError, parse_decision  # noqa: E402

START = {"messages": [{"role": "user", "text": "state"}]}


def final(qty, why="because"):
    return LLMReply(text=json.dumps({"order_qty": qty, "rationale": why}), input_tokens=10)


def tool(name="forecast_lookup"):
    return LLMReply(tool_calls=(ToolCall(name, {"k": 3}),))


TOOLS = {"forecast_lookup": lambda k: {"q90": 10 * k}}


def run(replies, cfg=None, tools=TOOLS):
    llm = ScriptedLLM(list(replies))
    return build_decision_graph(llm, tools, cfg).invoke(START), llm


def nodes(out):
    return [e["node"] for e in out["trace"]]


def test_direct_answer():
    out, llm = run([final(7)])
    assert (out["order_qty"], out["failure"]) == (7, None)
    assert nodes(out) == ["plan", "validate"] and len(llm.calls) == 1
    assert out["input_tokens"] == 10


def test_tool_loop_then_answer():
    out, _ = run([tool(), tool(), final(9)])
    assert out["tool_calls_used"] == 2 and out["order_qty"] == 9
    assert nodes(out).count("tools") == 2


def test_tool_cap_is_enforced():
    cfg = GraphConfig(max_tool_calls=1)
    # second tool request has tools_allowed=False -> schema failure -> retry -> answer
    out, llm = run([tool(), tool(), final(4)], cfg)
    assert out["tool_calls_used"] == 1 and out["order_qty"] == 4
    assert [c["tools_allowed"] for c in llm.calls] == [True, False, False]
    assert "retry" in nodes(out)


def test_c1_never_allows_tools():
    out, llm = run([final(3)], GraphConfig(max_tool_calls=0))
    assert llm.calls[0]["tools_allowed"] is False and out["order_qty"] == 3


def test_one_retry_then_fail_counts_as_failure():
    out, llm = run(
        [LLMReply(text="nope"), LLMReply(text="still no")], GraphConfig(failure_order_qty=5)
    )
    assert len(llm.calls) == 2  # exactly one retry
    assert out["failure"].startswith("schema") and out["order_qty"] == 5
    assert nodes(out)[-1] == "fail"


def test_retry_can_recover():
    out, _ = run([LLMReply(text="oops"), final(6)])
    assert out["failure"] is None and out["order_qty"] == 6


def test_guardrails_clip_and_escalate():
    cfg = GraphConfig(guardrails=True, max_order_qty=50)
    out, _ = run([final(500)], cfg)
    assert out["order_qty"] == 50 and out["escalate"] is True
    out, _ = run([final(20)], cfg)
    assert out["order_qty"] == 20 and out["escalate"] is False


def test_no_guardrails_passes_through():
    out, _ = run([final(500)], GraphConfig(guardrails=False, max_order_qty=50))
    assert out["order_qty"] == 500


@pytest.mark.parametrize(
    "text",
    [
        '{"order_qty": 1.5, "rationale": "x"}',
        '{"order_qty": true, "rationale": "x"}',
        '{"order_qty": -1, "rationale": "x"}',
        '{"order_qty": 1}',
        "[]",
        '{"order_qty": 1, "rationale": "x", "extra": 1}',
        '{"order_qty": 1, "rationale": " "}',
    ],
)
def test_schema_rejects(text):
    with pytest.raises(SchemaError):
        parse_decision(text)


def test_hosted_tracing_refused(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    with pytest.raises(RuntimeError):
        build_decision_graph(ScriptedLLM([]), {})
