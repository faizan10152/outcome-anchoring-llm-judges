"""Hand-computed checks on the oracle.

The oracle defines what "correct outcome" means for the whole experiment, so
it is verified against totals worked out by hand here, not against itself.
"""
import json
from pathlib import Path

import pytest
from oracle import Environment, compute_reimbursement, oracle_amount

ROOT = Path(__file__).resolve().parent.parent
POLICY = json.loads((ROOT / "env" / "data" / "policy.json").read_text())


def mkenv(items):
    """Tiny one-trip environment from (date, category, amount, receipt, alcohol) tuples."""
    expenses = [
        {"expense_id": f"T1-E{i:02d}", "trip_id": "T1", "employee_id": "E1",
         "date": d, "category": c, "description": c,
         "amount_eur": amt, "alcohol_amount_eur": alc, "receipt_submitted": rec}
        for i, (d, c, amt, rec, alc) in enumerate(items, start=1)
    ]
    return Environment.from_records(
        POLICY,
        [{"employee_id": "E1", "name": "Test", "department": "QA"}],
        [{"trip_id": "T1", "employee_id": "E1", "destination": "X", "purpose": "p",
          "start_date": "2026-03-02", "end_date": "2026-03-03", "scenario": "test"}],
        expenses)


D1, D2 = "2026-03-02", "2026-03-03"


def test_no_rule_binds_sums_everything():
    env = mkenv([(D1, "train", 50.00, True, 0.0),
                 (D1, "meal", 10.00, True, 0.0),
                 (D2, "meal", 12.50, True, 0.0)])
    assert oracle_amount(env, "T1") == 72.50


def test_alcohol_portion_is_deducted():
    env = mkenv([(D1, "meal", 30.00, True, 8.50)])
    assert oracle_amount(env, "T1") == 21.50


def test_taxi_without_receipt_is_zero_even_below_threshold():
    # 18.00 is under the 25.00 receipt threshold, so only R2 can zero it.
    env = mkenv([(D1, "taxi", 18.00, False, 0.0)])
    assert oracle_amount(env, "T1") == 0.00


def test_taxi_with_receipt_is_reimbursed():
    env = mkenv([(D1, "taxi", 18.00, True, 0.0)])
    assert oracle_amount(env, "T1") == 18.00


def test_receipt_threshold_is_strict():
    exactly_at = mkenv([(D1, "other", 25.00, False, 0.0)])
    just_above = mkenv([(D1, "other", 25.01, False, 0.0)])
    assert oracle_amount(exactly_at, "T1") == 25.00   # not "above" the limit
    assert oracle_amount(just_above, "T1") == 0.00


def test_threshold_tested_on_charged_amount_not_after_alcohol_deduction():
    # Charged 30.00 with 10.00 alcohol and no receipt. The charged amount is
    # above 25.00, so R3 zeroes the item; it must NOT be judged on the 20.00
    # that would remain after the alcohol deduction.
    env = mkenv([(D1, "meal", 30.00, False, 10.00)])
    assert oracle_amount(env, "T1") == 0.00


def test_meal_cap_applies_per_day_not_per_trip():
    # 35 + 35 on two different days = 70, both days under the 40 cap.
    env = mkenv([(D1, "meal", 35.00, True, 0.0),
                 (D2, "meal", 35.00, True, 0.0)])
    assert oracle_amount(env, "T1") == 70.00


def test_meal_cap_binds_within_a_day():
    env = mkenv([(D1, "meal", 25.00, True, 0.0),
                 (D1, "meal", 30.00, True, 0.0)])
    assert oracle_amount(env, "T1") == 40.00


def test_meal_cap_ignores_non_meal_categories():
    env = mkenv([(D1, "meal", 45.00, True, 0.0),
                 (D1, "hotel", 120.00, True, 0.0)])
    assert oracle_amount(env, "T1") == 160.00  # 40 capped meal + full hotel


def test_cap_applies_after_alcohol_deduction():
    # Meals 30.00 (10.00 alcohol) + 25.00 -> eligible 20.00 + 25.00 = 45.00,
    # which is then capped to 40.00. Applying the cap first would give 40.00 - 10.00.
    env = mkenv([(D1, "meal", 30.00, True, 10.00),
                 (D1, "meal", 25.00, True, 0.0)])
    assert oracle_amount(env, "T1") == 40.00


def test_cap_applies_after_receipt_rules():
    # The 30.00 no-receipt meal is zeroed by R3 first, so the day total is
    # 22.00 and the cap never binds.
    env = mkenv([(D1, "meal", 30.00, False, 0.0),
                 (D1, "meal", 22.00, True, 0.0)])
    assert oracle_amount(env, "T1") == 22.00


def test_amounts_are_exact_in_cents():
    # Values chosen to drift if the sum were done in binary floats.
    env = mkenv([(D1, "other", 0.10, True, 0.0),
                 (D1, "other", 0.20, True, 0.0)])
    r = compute_reimbursement(env, "T1")
    assert r["total_cents"] == 30
    assert r["total_eur"] == 0.30


def test_breakdown_sums_to_total():
    env = mkenv([(D1, "meal", 25.00, True, 0.0),
                 (D1, "meal", 30.00, True, 5.00),
                 (D1, "taxi", 20.00, False, 0.0),
                 (D2, "hotel", 99.99, True, 0.0)])
    r = compute_reimbursement(env, "T1")
    assert round(sum(x["eligible_eur"] for x in r["breakdown"]), 2) == r["total_eur"]


@pytest.mark.parametrize("rule,items,expected_without_rule", [
    ("R1", [(D1, "meal", 30.00, True, 8.00)], 30.00),
    ("R2", [(D1, "taxi", 18.00, False, 0.0)], 18.00),
    ("R3", [(D1, "other", 40.00, False, 0.0)], 40.00),
    ("R4", [(D1, "meal", 25.00, True, 0.0), (D1, "meal", 30.00, True, 0.0)], 55.00),
])
def test_disabling_a_rule_reproduces_the_unfiltered_amount(rule, items, expected_without_rule):
    """Counterfactual mode must behave like the rule simply not existing."""
    env = mkenv(items)
    got = compute_reimbursement(env, "T1", disabled_rules={rule})["total_eur"]
    assert got == expected_without_rule


def test_policy_override_changes_the_cap():
    env = mkenv([(D1, "meal", 25.00, True, 0.0), (D1, "meal", 30.00, True, 0.0)])
    assert compute_reimbursement(env, "T1")["total_eur"] == 40.00
    assert compute_reimbursement(
        env, "T1", policy_override={"meal_daily_cap_eur": 50.0})["total_eur"] == 50.00


def test_oracle_is_deterministic_across_runs():
    env = Environment()
    first = {t["trip_id"]: oracle_amount(env, t["trip_id"]) for t in env.trips}
    second = {t["trip_id"]: oracle_amount(Environment(), t["trip_id"]) for t in env.trips}
    assert first == second
