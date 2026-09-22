"""Ground-truth reimbursement calculation.

This is the oracle: the authoritative answer the agent is graded against.
It is deliberately independent of the agent and of the tool layer.

All arithmetic is done in whole cents (integers) so results are exact and
byte-stable across machines; only the public return values are floats.

Rules (see env/data/policy.json for the text shown to the agent):
    R1 alcohol        - deduct the alcohol portion of every item
    R2 taxi receipt   - taxi without a receipt reimburses 0
    R3 receipt limit  - any item charged above the threshold without a
                        receipt reimburses 0 (tested on the charged amount,
                        before the R1 deduction)
    R4 meal cap       - eligible meal spend per calendar day is capped
Applied in that order.

`compute_reimbursement` accepts `policy_override` and `disabled_rules` so the
same code can produce counterfactual amounts, e.g. "what the agent would have
submitted if it believed the meal cap were 50 EUR" or "...if it never applied
the taxi rule". The fault injector uses this to build outcome-wrong variants.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

DATA_DIR = Path(__file__).parent / "data"
ALL_RULES = ("R1", "R2", "R3", "R4")


def _cents(value: float) -> int:
    """Convert a EUR float with 2 decimals to whole cents, exactly."""
    return int(round(float(value) * 100))


def _eur(cents: int) -> float:
    return round(cents / 100, 2)


class Environment:
    """Read-only view over the generated JSON database."""

    def __init__(self, data_dir: Path | str = DATA_DIR):
        self.data_dir = Path(data_dir)
        self.policy = json.loads((self.data_dir / "policy.json").read_text())
        self.employees = json.loads((self.data_dir / "employees.json").read_text())
        self.trips = json.loads((self.data_dir / "trips.json").read_text())
        self.expenses = json.loads((self.data_dir / "expenses.json").read_text())
        self._emp_by_id = {e["employee_id"]: e for e in self.employees}
        self._trip_by_id = {t["trip_id"]: t for t in self.trips}
        self._exp_by_trip = defaultdict(list)
        for x in self.expenses:
            self._exp_by_trip[x["trip_id"]].append(x)

    @classmethod
    def from_records(cls, policy: dict, employees: list, trips: list,
                     expenses: list) -> "Environment":
        """Build an environment from in-memory records (used by the tests)."""
        obj = cls.__new__(cls)
        obj.data_dir = None
        obj.policy, obj.employees = policy, employees
        obj.trips, obj.expenses = trips, expenses
        obj._emp_by_id = {e["employee_id"]: e for e in employees}
        obj._trip_by_id = {t["trip_id"]: t for t in trips}
        obj._exp_by_trip = defaultdict(list)
        for x in expenses:
            obj._exp_by_trip[x["trip_id"]].append(x)
        return obj

    def employee(self, employee_id: str) -> dict | None:
        return self._emp_by_id.get(employee_id)

    def trip(self, trip_id: str) -> dict | None:
        return self._trip_by_id.get(trip_id)

    def trips_of(self, employee_id: str) -> list[dict]:
        return [t for t in self.trips if t["employee_id"] == employee_id]

    def expenses_of(self, trip_id: str) -> list[dict]:
        return list(self._exp_by_trip.get(trip_id, []))


def compute_reimbursement(
    env: Environment,
    trip_id: str,
    policy_override: dict | None = None,
    disabled_rules: Iterable[str] = (),
) -> dict:
    """Return the reimbursable total for one trip, plus a per-item breakdown.

    policy_override: keys of policy["rules"] to replace, e.g.
        {"meal_daily_cap_eur": 50.0} for the counterfactual behind a
        fabricated-cap fault.
    disabled_rules: rule ids the calculation should skip, e.g. {"R2"} for the
        counterfactual behind a skipped taxi-receipt check.
    """
    disabled = set(disabled_rules)
    rules = dict(env.policy["rules"])
    rules.update(policy_override or {})

    receipt_limit = _cents(rules["receipt_required_above_eur"])
    meal_cap = _cents(rules["meal_daily_cap_eur"])
    taxi_needs_receipt = bool(rules["taxi_requires_receipt"])
    alcohol_ok = bool(rules["alcohol_reimbursable"])

    breakdown = []
    for x in sorted(env.expenses_of(trip_id), key=lambda i: i["expense_id"]):
        charged = _cents(x["amount_eur"])
        alcohol = _cents(x["alcohol_amount_eur"])
        eligible, reason = charged, "reimbursed in full"

        # R1 - alcohol is not reimbursable
        if "R1" not in disabled and not alcohol_ok and alcohol > 0:
            eligible -= alcohol
            reason = f"alcohol portion {_eur(alcohol):.2f} EUR deducted"

        # R2 - taxi requires a receipt
        if ("R2" not in disabled and taxi_needs_receipt
                and x["category"] == "taxi" and not x["receipt_submitted"]):
            eligible, reason = 0, "taxi without receipt - not reimbursable"

        # R3 - receipt required above the threshold (tested on charged amount)
        elif ("R3" not in disabled and charged > receipt_limit
                and not x["receipt_submitted"]):
            eligible, reason = 0, (
                f"charged above {_eur(receipt_limit):.2f} EUR without receipt"
                " - not reimbursable")

        breakdown.append({
            "expense_id": x["expense_id"], "date": x["date"],
            "category": x["category"], "charged_eur": x["amount_eur"],
            "eligible_cents": eligible, "reason": reason,
        })

    # R4 - per-day meal cap, applied to what survived R1-R3
    capped_days = []
    if "R4" not in disabled:
        per_day = defaultdict(int)
        for row in breakdown:
            if row["category"] == "meal":
                per_day[row["date"]] += row["eligible_cents"]
        for day, total in sorted(per_day.items()):
            if total <= meal_cap:
                continue
            excess = total - meal_cap
            capped_days.append({
                "date": day, "meal_total_eur": _eur(total),
                "capped_to_eur": _eur(meal_cap), "excess_eur": _eur(excess),
            })
            # Reduce the day's meal rows (largest first) until the cap is met.
            meals = sorted([r for r in breakdown
                            if r["category"] == "meal" and r["date"] == day],
                           key=lambda r: -r["eligible_cents"])
            for row in meals:
                if excess <= 0:
                    break
                cut = min(excess, row["eligible_cents"])
                row["eligible_cents"] -= cut
                row["reason"] = f"reduced by daily meal cap ({day})"
                excess -= cut

    total = sum(r["eligible_cents"] for r in breakdown)
    for row in breakdown:
        row["eligible_eur"] = _eur(row.pop("eligible_cents"))

    return {
        "trip_id": trip_id,
        "total_eur": _eur(total),
        "total_cents": total,
        "breakdown": breakdown,
        "capped_days": capped_days,
        "policy_override": policy_override or {},
        "disabled_rules": sorted(disabled),
    }


def oracle_amount(env: Environment, trip_id: str) -> float:
    """The one number that defines a correct outcome for a trip."""
    return compute_reimbursement(env, trip_id)["total_eur"]


if __name__ == "__main__":
    env = Environment()
    for t in env.trips:
        r = compute_reimbursement(env, t["trip_id"])
        print(f"{t['trip_id']}  {t['employee_id']}  {t['destination']:<11}"
              f"  {t['scenario']:<26}  {r['total_eur']:>8.2f} EUR")
