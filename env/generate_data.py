"""Deterministic generator for the expense-reimbursement environment.

Produces employees.json, trips.json and expenses.json from a fixed seed so the
whole environment is reproducible from this repo alone.

Each trip is built from a *scenario* that controls which policy rules bind
(see SCENARIOS). Keeping the binding rules under explicit control matters for
the experiment: fault injection (e.g. a fabricated meal cap) only changes the
final reimbursed amount when the corresponding rule actually binds.

Run:  python env/generate_data.py
"""
from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
SEED = 20260922

EMPLOYEES = [
    ("Anna Weber", "Engineering"),
    ("Bernd Schulz", "Sales"),
    ("Clara Hoffmann", "Marketing"),
    ("David Krause", "Engineering"),
    ("Elena Fischer", "Finance"),
    ("Felix Braun", "Sales"),
    ("Greta Lange", "Legal"),
    ("Hannes Vogt", "Engineering"),
    ("Irina Sommer", "Marketing"),
    ("Jonas Keller", "Finance"),
    ("Katrin Busch", "Legal"),
    ("Lukas Mertens", "Sales"),
]

DESTINATIONS = [
    "Berlin", "Munich", "Hamburg", "Vienna", "Zurich", "Amsterdam",
    "Paris", "Cologne", "Frankfurt", "Brussels", "Copenhagen", "Prague",
]

# scenario -> how many trips use it. Rules that bind are encoded in the name.
# Only a small share of trips has no binding rule at all, because a trip where
# no rule binds cannot express most injected faults as a wrong final amount.
SCENARIOS = {
    "clean": 4,                      # no rule binds
    "meal_cap": 6,                   # R4
    "taxi_no_receipt": 5,            # R2
    "over_threshold_no_receipt": 5,  # R3
    "alcohol": 5,                    # R1
    "alcohol_and_cap": 4,            # R1 + R4
    "taxi_and_threshold": 4,         # R2 + R3
    "all_rules": 3,                  # R1 + R2 + R3 + R4
}

MEAL_LABELS = ["Breakfast", "Lunch", "Dinner"]


def eur(rng: random.Random, low: float, high: float) -> float:
    """Random amount in [low, high] rounded to whole cents."""
    return round(rng.uniform(low, high), 2)


def _expense(eid, tid, seq, category, description, day, amount,
             has_receipt=True, alcohol=0.0):
    return {
        "expense_id": f"{tid}-E{seq:02d}",
        "trip_id": tid,
        "employee_id": eid,
        "date": day.isoformat(),
        "category": category,
        "description": description,
        "amount_eur": round(amount, 2),
        "alcohol_amount_eur": round(alcohol, 2),
        "receipt_submitted": bool(has_receipt),
    }


def build_expenses(rng, eid, tid, start: date, nights: int, scenario: str):
    """Build the expense list for one trip under the given scenario."""
    days = [start + timedelta(days=i) for i in range(nights + 1)]
    items, seq = [], 1

    def add(**kw):
        nonlocal seq
        items.append(_expense(eid, tid, seq, **kw))
        seq += 1

    # --- travel + lodging (always compliant, keeps the task non-trivial) ---
    add(category="train", description="Train ticket (return)", day=days[0],
        amount=eur(rng, 45, 130), has_receipt=True)
    for night in range(nights):
        add(category="hotel", description=f"Hotel night {night + 1}", day=days[night],
            amount=eur(rng, 85, 155), has_receipt=True)

    # --- meals -----------------------------------------------------------
    cap_binds = scenario in ("meal_cap", "alcohol_and_cap", "all_rules")
    has_alcohol = scenario in ("alcohol", "alcohol_and_cap", "all_rules")

    for i, day in enumerate(days):
        # The cap binds on the first day only, so the correct answer requires
        # per-day grouping rather than a single total.
        if cap_binds and i == 0:
            targets = [eur(rng, 18, 26), eur(rng, 28, 38)]  # sums to 46-64
        else:
            targets = [eur(rng, 7, 14), eur(rng, 12, 22)]   # sums well under 40
        for j, amt in enumerate(targets):
            alcohol = 0.0
            # The alcohol item goes on the *last* day, never on the capped
            # first day: on a capped day the deduction is absorbed by the cap
            # and R1 would not bind at all (the two rules would mask each
            # other, and neither could be expressed as a wrong final amount).
            if has_alcohol and i == len(days) - 1 and j == len(targets) - 1:
                alcohol = min(eur(rng, 5.0, 11.0), round(amt * 0.5, 2))
            add(category="meal", description=f"{MEAL_LABELS[j % 3]} in transit" if i == 0
                else f"{MEAL_LABELS[j % 3]}", day=day, amount=amt,
                has_receipt=True, alcohol=alcohol)

    # --- taxi ------------------------------------------------------------
    if scenario in ("taxi_no_receipt", "taxi_and_threshold", "all_rules"):
        # Kept at or below the receipt threshold on purpose: above it, R3
        # would zero the same item and R2 would no longer bind on its own.
        add(category="taxi", description="Taxi to client office", day=days[0],
            amount=eur(rng, 9, 24), has_receipt=False)
    else:
        add(category="taxi", description="Taxi from station to hotel", day=days[0],
            amount=eur(rng, 9, 28), has_receipt=True)

    # --- an over-threshold item without a receipt -------------------------
    if scenario in ("over_threshold_no_receipt", "taxi_and_threshold", "all_rules"):
        add(category="other", description="Conference materials", day=days[-1],
            amount=eur(rng, 26.5, 75), has_receipt=False)
    elif rng.random() < 0.45:
        # A small no-receipt item that stays *under* the threshold, so the
        # receipt rule is present in the data without binding.
        add(category="other", description="Printing / stationery", day=days[-1],
            amount=eur(rng, 4, 19), has_receipt=False)

    return items


def main():
    rng = random.Random(SEED)

    employees = [
        {"employee_id": f"E{i + 1:03d}", "name": name, "department": dept}
        for i, (name, dept) in enumerate(EMPLOYEES)
    ]

    # 3 trips per employee, each to a different destination -> 36 trips.
    scenario_pool = [s for s, n in SCENARIOS.items() for _ in range(n)]
    rng.shuffle(scenario_pool)
    assert len(scenario_pool) == len(EMPLOYEES) * 3, len(scenario_pool)

    trips, expenses = [], []
    base_day = date(2026, 3, 2)

    for idx, emp in enumerate(employees):
        dests = rng.sample(DESTINATIONS, 3)
        for k, dest in enumerate(dests):
            tid = f"T{idx * 3 + k + 1:03d}"
            scenario = scenario_pool[idx * 3 + k]
            nights = rng.choice([1, 1, 2])
            start = base_day + timedelta(days=rng.randrange(0, 120))
            trips.append({
                "trip_id": tid,
                "employee_id": emp["employee_id"],
                "destination": dest,
                "purpose": rng.choice([
                    "Client meeting", "Conference", "Team workshop",
                    "Supplier audit", "Trade fair", "Project kickoff",
                ]),
                "start_date": start.isoformat(),
                "end_date": (start + timedelta(days=nights)).isoformat(),
                "scenario": scenario,
            })
            expenses += build_expenses(rng, emp["employee_id"], tid, start, nights, scenario)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, payload in [("employees", employees), ("trips", trips), ("expenses", expenses)]:
        path = DATA_DIR / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {path.relative_to(Path.cwd())}  ({len(payload)} records)")


if __name__ == "__main__":
    main()
