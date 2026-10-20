"""Strict output schema for the planner: {"order_qty": int >= 0, "rationale": str} (DRAFT)."""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    order_qty: int
    rationale: str


class SchemaError(ValueError):
    pass


def parse_decision(text: str) -> Decision:
    """Parse and strictly validate the model's JSON. Any deviation raises SchemaError."""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise SchemaError(f"not valid JSON: {exc}") from exc
    if not isinstance(obj, dict) or set(obj) != {"order_qty", "rationale"}:
        raise SchemaError("expected exactly the keys order_qty and rationale")
    qty, why = obj["order_qty"], obj["rationale"]
    if isinstance(qty, bool) or not isinstance(qty, int):
        raise SchemaError("order_qty must be an integer")
    if qty < 0:
        raise SchemaError("order_qty must be >= 0")
    if not isinstance(why, str) or not why.strip():
        raise SchemaError("rationale must be a non-empty string")
    return Decision(order_qty=qty, rationale=why)
