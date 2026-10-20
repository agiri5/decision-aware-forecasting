"""One-decision LangGraph for the planner agent (DRAFT topology v0, not frozen).

Topology (file 03, section E). Conditions differ only in configuration:
  C1  max_tool_calls = 0, guardrails off
  C2  max_tool_calls = 3, guardrails off
  C3  max_tool_calls = 3, guardrails on (order bounds, sanity clip, escalation flag)

    plan --tool request--> tools --> plan          (at most `max_tool_calls` calls)
    plan --final text----> validate --ok--> [guardrails] --> END
                              |-- invalid, attempt < 1 --> plan   (one retry max)
                              |-- invalid, retry spent --> fail --> END (counts as a failure)

No hidden LLM calls: every LLM call is `plan`. Tool functions are injected. The trace is local
(list of dicts, written to JSONL by the runner); no hosted tracing is used.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from agent.llm import LLMClient
from agent.schema import SchemaError, parse_decision

MAX_RETRIES = 1  # file 03: "one retry max"
_TRACING_VARS = ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING")


@dataclass(frozen=True)
class GraphConfig:
    max_tool_calls: int = 3
    guardrails: bool = False
    max_order_qty: int = 10_000  # C3 order bound (SYNTHETIC)
    failure_order_qty: int = 0  # order placed when output stays invalid. OPEN: pre-register this.


class DecisionState(TypedDict, total=False):
    messages: list[dict[str, Any]]
    tool_calls_used: int
    attempt: int
    raw_text: str
    order_qty: int
    rationale: str
    failure: str | None
    escalate: bool
    input_tokens: int
    output_tokens: int
    trace: list[dict[str, Any]]


def _refuse_hosted_tracing() -> None:
    on = [v for v in _TRACING_VARS if os.environ.get(v, "").lower() in {"1", "true", "yes"}]
    if on:
        raise RuntimeError(f"hosted tracing is disabled for this project; unset {on}")


def build_decision_graph(
    llm: LLMClient, tools: dict[str, Callable[..., Any]], cfg: GraphConfig | None = None
):
    """Compile the per-decision graph. `invoke({"messages": [...]})` runs one decision."""
    cfg = cfg or GraphConfig()
    _refuse_hosted_tracing()

    def plan(s: DecisionState) -> DecisionState:
        allowed = s.get("tool_calls_used", 0) < cfg.max_tool_calls
        reply = llm.complete(s["messages"], tools_allowed=allowed)
        msgs = [
            *s["messages"],
            {"role": "assistant", "text": reply.text, "calls": reply.tool_calls},
        ]
        wants_tool = bool(reply.tool_calls)
        event = {"node": "plan", "tools_allowed": allowed, "tool_request": wants_tool}
        return {
            "messages": msgs,
            "raw_text": "" if wants_tool else reply.text,
            "input_tokens": s.get("input_tokens", 0) + reply.input_tokens,
            "output_tokens": s.get("output_tokens", 0) + reply.output_tokens,
            "trace": [*s.get("trace", []), event],
            "failure": (
                "tool request beyond budget" if wants_tool and not allowed else s.get("failure")
            ),
        }

    def route_after_plan(s: DecisionState) -> str:
        last = s["messages"][-1]
        if last["calls"]:
            allowed = s.get("tool_calls_used", 0) < cfg.max_tool_calls
            return "tools" if allowed else "validate"  # over-budget request -> schema failure
        return "validate"

    def run_tools(s: DecisionState) -> DecisionState:
        used = s.get("tool_calls_used", 0)
        msgs, events = list(s["messages"]), []
        for call in s["messages"][-1]["calls"]:
            if used >= cfg.max_tool_calls:
                result: Any = "tool budget exhausted"
            elif call.name not in tools:
                result = f"unknown tool {call.name}"
            else:
                result = tools[call.name](**call.args)
                used += 1
            msgs.append({"role": "tool", "name": call.name, "result": result})
            events.append({"node": "tools", "tool": call.name})
        return {"messages": msgs, "tool_calls_used": used, "trace": [*s.get("trace", []), *events]}

    def validate(s: DecisionState) -> DecisionState:
        trace = s.get("trace", [])
        try:
            if not s.get("raw_text"):
                raise SchemaError("no final answer")
            d = parse_decision(s["raw_text"])
        except SchemaError as exc:
            return {
                "failure": f"schema: {exc}",
                "trace": [*trace, {"node": "validate", "ok": False, "error": str(exc)}],
            }
        return {
            "order_qty": d.order_qty,
            "rationale": d.rationale,
            "failure": None,
            "trace": [*trace, {"node": "validate", "ok": True}],
        }

    def route_after_validate(s: DecisionState) -> str:
        if s.get("failure") is None:
            return "guardrails" if cfg.guardrails else END
        return "retry" if s.get("attempt", 0) < MAX_RETRIES else "fail"

    def retry(s: DecisionState) -> DecisionState:
        note = {"role": "user", "text": f"Invalid output ({s['failure']}). Reply with valid JSON."}
        return {
            "attempt": s.get("attempt", 0) + 1,
            "failure": None,
            "messages": [*s["messages"], note],
            "trace": [*s.get("trace", []), {"node": "retry"}],
        }

    def fail(s: DecisionState) -> DecisionState:
        return {
            "order_qty": cfg.failure_order_qty,
            "rationale": "",
            "trace": [*s.get("trace", []), {"node": "fail", "reason": s["failure"]}],
        }

    def guardrails(s: DecisionState) -> DecisionState:
        qty = min(max(s["order_qty"], 0), cfg.max_order_qty)
        clipped = qty != s["order_qty"]
        return {
            "order_qty": qty,
            "escalate": clipped,
            "trace": [*s.get("trace", []), {"node": "guardrails", "clipped": clipped}],
        }

    g = StateGraph(DecisionState)
    for name, fn in [
        ("plan", plan),
        ("tools", run_tools),
        ("validate", validate),
        ("retry", retry),
        ("fail", fail),
        ("guardrails", guardrails),
    ]:
        g.add_node(name, fn)
    g.set_entry_point("plan")
    g.add_conditional_edges("plan", route_after_plan, {"tools": "tools", "validate": "validate"})
    g.add_edge("tools", "plan")
    g.add_conditional_edges(
        "validate",
        route_after_validate,
        {"guardrails": "guardrails", END: END, "retry": "retry", "fail": "fail"},
    )
    g.add_edge("retry", "plan")
    g.add_edge("fail", END)
    g.add_edge("guardrails", END)
    return g.compile()
