"""Integrity + design checks on the generated environment.

Beyond data sanity this answers the question that matters for the experiment:
for each trip, which policy rules actually *bind* (i.e. change the reimbursed
amount)? A rule that does not bind cannot be turned into a wrong final outcome
by fault injection, so the scenario labels must be backed by real bindings.

Run:  python env/validate.py
"""
from __future__ import annotations

import sys
from collections import Counter

from oracle import ALL_RULES, Environment, compute_reimbursement

EXPECTED_BINDINGS = {
    "clean": set(),
    "meal_cap": {"R4"},
    "taxi_no_receipt": {"R2"},
    "over_threshold_no_receipt": {"R3"},
    "alcohol": {"R1"},
    "alcohol_and_cap": {"R1", "R4"},
    "taxi_and_threshold": {"R2", "R3"},
    "all_rules": {"R1", "R2", "R3", "R4"},
}


def binding_rules(env, trip_id) -> set[str]:
    """A rule binds if disabling it changes the reimbursed total."""
    base = compute_reimbursement(env, trip_id)["total_cents"]
    return {r for r in ALL_RULES
            if compute_reimbursement(env, trip_id, disabled_rules={r})["total_cents"] != base}


def main() -> int:
    env = Environment()
    errors, warnings = [], []

    # --- referential integrity + field sanity ---------------------------
    emp_ids = {e["employee_id"] for e in env.employees}
    trip_ids = {t["trip_id"] for t in env.trips}
    if len(emp_ids) != len(env.employees):
        errors.append("duplicate employee_id")
    if len(trip_ids) != len(env.trips):
        errors.append("duplicate trip_id")

    seen_exp = set()
    for x in env.expenses:
        xid = x["expense_id"]
        if xid in seen_exp:
            errors.append(f"duplicate expense_id {xid}")
        seen_exp.add(xid)
        if x["trip_id"] not in trip_ids:
            errors.append(f"{xid}: unknown trip {x['trip_id']}")
        if x["employee_id"] not in emp_ids:
            errors.append(f"{xid}: unknown employee {x['employee_id']}")
        if x["amount_eur"] <= 0:
            errors.append(f"{xid}: non-positive amount")
        if x["alcohol_amount_eur"] < 0 or x["alcohol_amount_eur"] > x["amount_eur"]:
            errors.append(f"{xid}: alcohol portion out of range")
        if round(x["amount_eur"], 2) != x["amount_eur"]:
            errors.append(f"{xid}: amount not in whole cents")
        if x["category"] not in env.policy["expense_categories"]:
            errors.append(f"{xid}: unknown category {x['category']}")

    for t in env.trips:
        if t["employee_id"] not in emp_ids:
            errors.append(f"{t['trip_id']}: unknown employee")
        if not env.expenses_of(t["trip_id"]):
            errors.append(f"{t['trip_id']}: no expenses")
        if t["start_date"] > t["end_date"]:
            errors.append(f"{t['trip_id']}: end before start")

    # employee+destination must be unique, so "Anna's Paris trip" is unambiguous
    pairs = Counter((t["employee_id"], t["destination"]) for t in env.trips)
    for pair, n in pairs.items():
        if n > 1:
            errors.append(f"ambiguous task reference: {pair} appears {n}x")

    # --- do the intended rules actually bind? ---------------------------
    bind_counter = Counter()
    for t in env.trips:
        actual = binding_rules(env, t["trip_id"])
        expected = EXPECTED_BINDINGS[t["scenario"]]
        bind_counter.update(actual or {"<none>"})
        if actual != expected:
            errors.append(
                f"{t['trip_id']} ({t['scenario']}): binds {sorted(actual) or '[]'}, "
                f"expected {sorted(expected) or '[]'}")

    # --- counterfactual separation --------------------------------------
    # A fabricated meal cap (40 -> 50) must move the answer wherever R4 binds;
    # this is what makes an outcome-wrong variant available for that fault.
    movable = 0
    for t in env.trips:
        base = compute_reimbursement(env, t["trip_id"])["total_cents"]
        alt = compute_reimbursement(env, t["trip_id"],
                                    policy_override={"meal_daily_cap_eur": 50.0})["total_cents"]
        r4_binds = "R4" in EXPECTED_BINDINGS[t["scenario"]]
        if r4_binds and alt == base:
            errors.append(f"{t['trip_id']}: meal cap binds but 50 EUR cap does not move the total")
        if alt != base:
            movable += 1

    # every trip needs at least one binding rule to carry an outcome-wrong twin
    no_binding = [t["trip_id"] for t in env.trips
                  if not EXPECTED_BINDINGS[t["scenario"]]]
    if len(no_binding) > len(env.trips) * 0.2:
        warnings.append(f"{len(no_binding)} trips have no binding rule: {no_binding}")

    # --- report ----------------------------------------------------------
    print(f"employees={len(env.employees)}  trips={len(env.trips)}  expenses={len(env.expenses)}")
    print(f"expenses per trip: min={min(len(env.expenses_of(t['trip_id'])) for t in env.trips)} "
          f"max={max(len(env.expenses_of(t['trip_id'])) for t in env.trips)}")
    print(f"rule bindings across trips: {dict(bind_counter)}")
    print(f"trips whose total moves under a fabricated 50 EUR meal cap: {movable}/{len(env.trips)}")
    print(f"trips with no binding rule (clean scenario): {len(no_binding)}")

    for w in warnings:
        print(f"WARN  {w}")
    for e in errors:
        print(f"FAIL  {e}")
    print("\nVALIDATION:", "PASSED" if not errors else f"FAILED ({len(errors)} errors)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
