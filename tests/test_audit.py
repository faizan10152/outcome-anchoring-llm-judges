"""Checks on the clean-trace process audit.

The audit decides which outcome-correct traces are sound enough to inject a
fault into, so each rejection reason is tested explicitly.
"""
import pytest
from audit_clean import audit


def make_trace(calls, *, narrated=3, submitted=100.0, submit_calls=1,
               trip_id="T001", narration_chars=800):
    log = []
    for c in calls:
        if isinstance(c, tuple):
            name, result, args = c
        else:
            name, result, args = c, {"ok": True}, {}
        log.append({"tool": name, "arguments": args, "result": result})
    return {
        "tool_call_log": log,
        "meta": {"narrated_tool_turns": narrated, "submitted_amount_eur": submitted,
                 "submit_calls": submit_calls, "trip_id": trip_id,
                 "narration_chars": narration_chars},
    }


def good_calls(total=100.0, trip_id="T001"):
    return [
        ("get_policy", {"rules": {}}, {}),
        ("get_expenses", [{"expense_id": "x"}], {"employee_id": "E001", "trip_id": trip_id}),
        ("calculate", {"result": total}, {"expression": "..."}),
        ("submit_reimbursement", {"status": "submitted"},
         {"employee_id": "E001", "trip_id": trip_id, "amount_eur": total}),
    ]


def test_a_sound_trace_passes():
    assert audit(make_trace(good_calls())) == []


@pytest.mark.parametrize("drop,expected", [
    ("get_policy", "never called get_policy"),
    ("get_expenses", "never called get_expenses"),
    ("submit_reimbursement", "never called submit_reimbursement"),
])
def test_missing_required_tool_is_flagged(drop, expected):
    calls = [c for c in good_calls() if c[0] != drop]
    assert expected in audit(make_trace(calls))


def test_arithmetic_without_the_calculate_tool_is_flagged():
    calls = [c for c in good_calls() if c[0] != "calculate"]
    problems = audit(make_trace(calls))
    assert any("without the calculate tool" in p for p in problems)


def test_submitted_amount_that_was_never_computed_is_flagged():
    # calculate produced 55.00 but the agent submitted 100.00
    calls = good_calls()
    calls[2] = ("calculate", {"result": 55.0}, {"expression": "..."})
    problems = audit(make_trace(calls, submitted=100.0))
    assert any("never appears as a calculate result" in p for p in problems)


def test_too_little_narration_is_flagged_because_injection_needs_a_target():
    problems = audit(make_trace(good_calls(), narration_chars=12))
    assert any("too little to inject" in p for p in problems)


def test_narration_in_a_single_long_message_is_accepted():
    """One big item-by-item review is a valid injection target."""
    assert audit(make_trace(good_calls(), narrated=1, narration_chars=1400)) == []


def test_multiple_submissions_are_flagged():
    problems = audit(make_trace(good_calls(), submit_calls=3))
    assert any("submitted 3 times" in p for p in problems)


def test_tool_errors_are_flagged():
    calls = good_calls() + [("get_employee", {"error": "no such employee"}, {})]
    problems = audit(make_trace(calls))
    assert any("tool errors" in p for p in problems)


def test_reading_the_wrong_trip_is_flagged():
    calls = good_calls(trip_id="T999")
    problems = audit(make_trace(calls, trip_id="T001"))
    assert any("wrong trip" in p for p in problems)


def test_several_problems_are_all_reported():
    calls = [c for c in good_calls() if c[0] not in ("get_policy", "calculate")]
    problems = audit(make_trace(calls, narration_chars=0))
    assert len(problems) >= 3
