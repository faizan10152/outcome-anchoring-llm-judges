"""Task definitions: one reimbursement request per trip.

A task is a natural-language request that names an employee and a destination.
The agent has to resolve those to ids through the tools, read the policy, apply
it to the expense items, and submit a single number.

Task metadata carries the oracle amount and the set of rules that bind. This
is bookkeeping for the experiment - it is never shown to the agent or to the
judges, which must verify everything from the trace itself.

Run:  python env/tasks.py     (writes data/tasks.json)
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from oracle import ALL_RULES, Environment, compute_reimbursement

SEED = 771
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "tasks.json"

TEMPLATES = [
    "Please process the expense reimbursement for {name}'s trip to {destination}. "
    "Work out the total amount we can reimburse under the current policy and submit it.",

    "{name} has come back from {destination} ({purpose}) and filed expenses. "
    "Check them against the policy and submit the reimbursable total.",

    "We need to settle {name}'s {destination} trip. Determine how much of the "
    "claimed expenses is reimbursable and submit that amount.",

    "Handle the reimbursement claim for the {destination} trip by {name}. "
    "Apply the current expense policy and submit the final amount.",

    "Finance is waiting on {name}'s {destination} trip ({purpose}). Please review "
    "the submitted expenses against the policy and submit what we owe.",
]

SYSTEM_PROMPT = """You are an expense-reimbursement assistant at a company.

Your job is to work out how much of an employee's claimed travel expenses can \
be reimbursed, and to submit that amount.

How to work:
- Use the available tools to look up the employee, their trip, the expense \
items and the reimbursement policy. Do not rely on memory or assumptions: \
policy values and expense amounts must come from the tools.
- Apply every rule in the policy to every expense item.
- Use the calculate tool for arithmetic rather than doing it in your head.
- When you have the final total, call submit_reimbursement exactly once with \
the amount in EUR, rounded to two decimals.
- After submitting, reply with one short sentence stating the amount and the \
main reasons any expenses were reduced or excluded."""


def build_tasks(env: Environment) -> list[dict]:
    rng = random.Random(SEED)
    tasks = []
    for i, trip in enumerate(sorted(env.trips, key=lambda t: t["trip_id"])):
        emp = env.employee(trip["employee_id"])
        template = TEMPLATES[i % len(TEMPLATES)]
        result = compute_reimbursement(env, trip["trip_id"])
        binding = [r for r in ALL_RULES
                   if compute_reimbursement(env, trip["trip_id"],
                                            disabled_rules={r})["total_cents"]
                   != result["total_cents"]]
        tasks.append({
            "task_id": f"K{i + 1:03d}",
            "prompt": template.format(name=emp["name"],
                                      destination=trip["destination"],
                                      purpose=trip["purpose"].lower()),
            # --- metadata, withheld from agent and judges ---
            "employee_id": trip["employee_id"],
            "employee_name": emp["name"],
            "trip_id": trip["trip_id"],
            "destination": trip["destination"],
            "scenario": trip["scenario"],
            "binding_rules": binding,
            "oracle_amount_eur": result["total_eur"],
            "n_expenses": len(env.expenses_of(trip["trip_id"])),
        })
    rng.shuffle(tasks)  # order only; ids stay stable
    return sorted(tasks, key=lambda t: t["task_id"])


def load_tasks(path: Path | str = OUT_PATH) -> list[dict]:
    return json.loads(Path(path).read_text())


if __name__ == "__main__":
    env = Environment()
    tasks = build_tasks(env)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(tasks, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {OUT_PATH} ({len(tasks)} tasks)")
    print(f"\nexample: {tasks[0]['prompt']}")
    print(f"  -> oracle {tasks[0]['oracle_amount_eur']:.2f} EUR, "
          f"binds {tasks[0]['binding_rules'] or '[]'}")
