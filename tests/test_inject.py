"""Checks on fault injection and matched-pair construction.

The experiment's causal claim depends on these invariants, so they are tested on
synthetic traces as well as verified on the real ones by faults/verify_pairs.py.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "faults"))
from inject import (build_expression, delete_call, edit_amount_in_analysis,  # noqa: E402
                    fabricate, find_analysis_index, find_call, fmt)

ANALYSIS = """I will now analyze each expense item.

1. **Expense ID: T001-E01**
   - **Date:** 2026-04-02
   - **Category:** train
   - **Charged amount:** 108.90 EUR
   - **Reimbursement status:** Reimbursed in full
   - **Policy rule:** No rule applies to train expenses.

2. **Expense ID: T001-E02**
   - **Date:** 2026-04-02
   - **Category:** hotel
   - **Charged Amount:** 104.58 EUR
   - **Reimbursement Status:** Reimbursed in full
"""


def msgs():
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "get_policy", "arguments": {}}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "get_policy", "content": "{}"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c2", "type": "function",
             "function": {"name": "get_expenses", "arguments": {}}}]},
        {"role": "tool", "tool_call_id": "c2", "name": "get_expenses", "content": "[]"},
        {"role": "user", "content": "Before doing any arithmetic, write out your analysis."},
        {"role": "assistant", "content": ANALYSIS},
    ]


def test_analysis_message_is_located():
    assert find_analysis_index(msgs()) == 7


def test_analysis_index_is_none_when_absent():
    assert find_analysis_index([{"role": "user", "content": "hi"}]) is None


def test_call_is_located_with_its_reply():
    assert find_call(msgs(), "get_policy") == (2, 3)
    assert find_call(msgs(), "get_expenses") == (4, 5)
    assert find_call(msgs(), "calculate") is None


@pytest.mark.parametrize("expense,amount,label", [
    ("T001-E01", 108.90, "Charged amount"),
    ("T001-E02", 104.58, "Charged Amount"),   # capitalisation varies by trace
])
def test_amount_edit_targets_the_right_block(expense, amount, label):
    out = edit_amount_in_analysis(ANALYSIS, expense, amount, 78.90)
    assert out is not None
    new_text, before, after = out
    assert "78.90 EUR" in new_text
    assert f"{amount:.2f} EUR" not in new_text.split(expense)[1][:200]
    # the *other* item must be untouched
    other = "104.58" if expense == "T001-E01" else "108.90"
    assert other in new_text


def test_amount_edit_refuses_when_the_amount_does_not_match():
    """A mismatch means the block was misparsed, so it must not be edited."""
    assert edit_amount_in_analysis(ANALYSIS, "T001-E01", 999.99, 78.90) is None


def test_amount_edit_refuses_unknown_expense():
    assert edit_amount_in_analysis(ANALYSIS, "T099-E99", 108.90, 78.90) is None


def test_deleting_a_call_removes_both_messages_and_stays_wellformed():
    out = delete_call(msgs(), "get_policy")
    assert out is not None
    trimmed, at = out
    assert at == 2 and len(trimmed) == len(msgs()) - 2
    names = [c["function"]["name"] for m in trimmed if m["role"] == "assistant"
             for c in m.get("tool_calls") or []]
    assert "get_policy" not in names
    ids = {c["id"] for m in trimmed if m["role"] == "assistant"
           for c in m.get("tool_calls") or []}
    assert all(m["tool_call_id"] in ids for m in trimmed if m["role"] == "tool")


def test_deleting_a_missing_call_is_refused():
    assert delete_call(msgs(), "calculate") is None


def test_the_analysis_survives_deletion():
    """F2/F3 delete a lookup but must keep the claims that make it a fault."""
    trimmed, _ = delete_call(msgs(), "get_expenses")
    assert any("Charged amount" in (m.get("content") or "") for m in trimmed)


@pytest.mark.parametrize("amount", [108.90, 40.0, 61.0, 12.5])
def test_fabricated_amount_differs_enough_to_be_visible(amount):
    fake = fabricate(amount)
    assert abs(fake - amount) >= 20.0
    assert fake > 0


def test_expression_formatting_matches_the_agent_style():
    assert fmt(108.90) == "108.9"
    assert fmt(12.00) == "12"
    assert build_expression([108.9, 12.0, 0.0]) == "108.9 + 12"


def test_expression_evaluates_to_the_intended_total():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "env"))
    from oracle import Environment
    from tools import ToolSession
    session = ToolSession(Environment())
    amounts = [108.9, 104.58, 12.38, 9.86]
    got = session.calculate(build_expression(amounts))
    assert abs(got["result"] - sum(amounts)) < 0.005
