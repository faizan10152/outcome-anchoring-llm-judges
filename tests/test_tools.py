"""Checks on the tool layer the agent talks to.

Two things matter here: tools must be faithful to the database (a judge
reading the trace has to be able to trust tool outputs as ground truth), and
they must fail softly, since a raised exception would truncate a trace.
"""
import pytest
from oracle import Environment
from tools import TOOL_NAMES, TOOL_SCHEMAS, ToolSession


@pytest.fixture
def session():
    return ToolSession(Environment())


def test_every_schema_has_an_implementation(session):
    for name in TOOL_NAMES:
        assert callable(getattr(session, name, None)), f"{name} has a schema but no method"


def test_schemas_are_wellformed():
    for s in TOOL_SCHEMAS:
        fn = s["function"]
        assert s["type"] == "function"
        assert fn["description"].strip()
        params = fn["parameters"]
        assert params["type"] == "object"
        for req in params["required"]:
            assert req in params["properties"], f"{fn['name']}: required field {req} not declared"


def test_get_expenses_matches_the_database(session):
    env = session.env
    for trip in env.trips:
        got = session.get_expenses(trip["employee_id"], trip["trip_id"])
        assert isinstance(got, list) and got
        assert len(got) == len(env.expenses_of(trip["trip_id"]))
        by_id = {x["expense_id"]: x for x in env.expenses_of(trip["trip_id"])}
        for row in got:
            src = by_id[row["expense_id"]]
            assert row["amount_eur"] == src["amount_eur"]
            assert row["receipt_submitted"] == src["receipt_submitted"]
            assert row["alcohol_amount_eur"] == src["alcohol_amount_eur"]


def test_policy_tool_exposes_the_values_the_rules_need(session):
    rules = session.get_policy()["rules"]
    assert rules["meal_daily_cap_eur"] == 40.0
    assert rules["receipt_required_above_eur"] == 25.0
    assert rules["taxi_requires_receipt"] is True
    assert rules["alcohol_reimbursable"] is False


@pytest.mark.parametrize("call,args", [
    ("get_employee", {"employee_id": "E999"}),
    ("get_expenses", {"employee_id": "E001", "trip_id": "T999"}),
    ("get_expenses", {"employee_id": "E002", "trip_id": "T001"}),  # wrong owner
    ("submit_reimbursement", {"employee_id": "E001", "trip_id": "T999", "amount_eur": 10}),
    ("submit_reimbursement", {"employee_id": "E001", "trip_id": "T001", "amount_eur": "abc"}),
])
def test_bad_input_returns_an_error_instead_of_raising(session, call, args):
    result = getattr(session, call)(**args)
    assert isinstance(result, dict) and "error" in result


def test_unknown_tool_and_bad_arguments_are_reported(session):
    assert "error" in session.call("delete_everything", {})
    assert "error" in session.call("get_employee", {"wrong_kwarg": 1})


@pytest.mark.parametrize("expr,expected", [
    ("12.50 + 30.00", 42.5),
    ("2 * (3 + 4)", 14.0),
    ("min(45.20, 40.00)", 40.0),
    ("round(10.005 * 2, 2)", 20.01),
    ("-5 + 10", 5.0),
])
def test_calculate_arithmetic(session, expr, expected):
    assert session.calculate(expr)["result"] == expected


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo hi')",
    "open('/etc/passwd').read()",
    "[1,2,3]",
    "1/0",
    "not_a_number",
    "",
])
def test_calculate_rejects_anything_that_is_not_plain_arithmetic(session, expr):
    assert "error" in session.calculate(expr)


def test_submission_is_recorded_and_rounded(session):
    assert session.submission is None
    session.call("submit_reimbursement",
                 {"employee_id": "E001", "trip_id": "T001", "amount_eur": 270.909})
    assert session.submission == {"employee_id": "E001", "trip_id": "T001",
                                  "amount_eur": 270.91}


def test_call_log_records_every_call(session):
    session.call("get_policy", {})
    session.call("get_employee", {"employee_id": "E001"})
    assert [c["tool"] for c in session.call_log] == ["get_policy", "get_employee"]
    assert all("result" in c for c in session.call_log)


def test_tools_do_not_leak_the_oracle_answer(session):
    """The agent (and later the judge) must not be handed the correct total."""
    import json
    from oracle import oracle_amount
    blob = json.dumps([session.get_employee("E001"), session.get_policy(),
                       session.get_expenses("E001", "T001")])
    assert f"{oracle_amount(session.env, 'T001'):.2f}" not in blob
    assert "scenario" not in blob
