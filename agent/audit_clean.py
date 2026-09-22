"""Process audit for clean traces.

`run_agent.py` sorts traces by *outcome*: did the submitted amount match the
oracle? That is not sufficient. A trace can reach the right number through a
bad process - never reading the policy, never doing the arithmetic, or simply
guessing well. Such a trace is unusable as the base for fault injection:

  - injecting F3 ("skipped the policy check") into a trace that never checked
    the policy injects nothing;
  - injecting F1 ("states a value the tool never returned") needs the agent to
    have stated tool values in the first place;
  - a trace whose submitted amount was never computed is already faulty, so an
    injected fault would be confounded with a pre-existing one.

This script flags those. It is a gate, not a substitute for reading traces by
hand (see the quality checks in the README).

    python agent/audit_clean.py
    python agent/audit_clean.py --move-rejected   # push failures to traces/failed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "env"))

import yaml  # noqa: E402

REQUIRED_TOOLS = ("get_policy", "get_expenses", "submit_reimbursement")
# Enough stated text that a fabricated claim can be edited in naturally.
MIN_NARRATION_CHARS = 200


def audit(trace: dict) -> list[str]:
    """Return a list of process problems; empty means the trace is usable."""
    problems = []
    m, log = trace["meta"], trace["tool_call_log"]
    called = [c["tool"] for c in log]

    for tool in REQUIRED_TOOLS:
        if tool not in called:
            problems.append(f"never called {tool}")

    if "calculate" not in called:
        problems.append("did the arithmetic without the calculate tool")

    if m["submit_calls"] > 1:
        problems.append(f"submitted {m['submit_calls']} times")

    # The submitted total should be something the agent actually computed.
    submitted = m["submitted_amount_eur"]
    calc_results = [c["result"].get("result") for c in log
                    if c["tool"] == "calculate" and isinstance(c["result"], dict)
                    and "result" in c["result"]]
    if submitted is not None and calc_results:
        if not any(abs(float(v) - submitted) < 0.005 for v in calc_results):
            problems.append("submitted amount never appears as a calculate result")

    # Injection targets: F1 and F2 need claims in assistant text to edit.
    # Volume matters more than which turn it landed on - this agent tends to
    # write one long item-by-item review rather than narrating every call.
    if m.get("narration_chars", 0) < MIN_NARRATION_CHARS:
        problems.append(f"only {m.get('narration_chars', 0)} chars of narration "
                        f"(< {MIN_NARRATION_CHARS}; too little to inject a claim into)")

    # A tool call whose result was an error suggests a confused episode.
    errors = [c["tool"] for c in log
              if isinstance(c["result"], dict) and "error" in c["result"]]
    if errors:
        problems.append(f"tool errors during the run: {sorted(set(errors))}")

    # The expenses of the right trip must have been read.
    got = [c for c in log if c["tool"] == "get_expenses"]
    if got and not any(c["arguments"].get("trip_id") == m["trip_id"] for c in got):
        problems.append("read expenses for the wrong trip")

    return problems


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config" / "experiment.yaml").read_text())
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--move-rejected", action="store_true",
                    help="move traces that fail the audit into the failed/ directory")
    args = ap.parse_args()

    clean_dir = ROOT / cfg["paths"]["traces_clean"]
    failed_dir = ROOT / cfg["paths"]["traces_failed"]
    files = sorted(clean_dir.glob("*.json"))
    if not files:
        print(f"no clean traces in {clean_dir.relative_to(ROOT)}")
        return 0

    ok, rejected = 0, []
    print(f"{'trace':<24}{'narr':>6}{'calls':>7}  verdict")
    for f in files:
        trace = json.loads(f.read_text())
        problems = audit(trace)
        m = trace["meta"]
        verdict = "usable" if not problems else "; ".join(problems)
        print(f"{f.stem:<24}{m['narrated_tool_turns']:>6}{m['tool_calls']:>7}  {verdict}")
        if problems:
            rejected.append(f)
        else:
            ok += 1

    print(f"\nusable for injection: {ok}/{len(files)}")
    if rejected and args.move_rejected:
        failed_dir.mkdir(parents=True, exist_ok=True)
        for f in rejected:
            f.rename(failed_dir / f.name)
        print(f"moved {len(rejected)} rejected trace(s) to {failed_dir.relative_to(ROOT)}")
    elif rejected:
        print("re-run these tasks, or pass --move-rejected to set them aside")
    return 0


if __name__ == "__main__":
    sys.exit(main())
