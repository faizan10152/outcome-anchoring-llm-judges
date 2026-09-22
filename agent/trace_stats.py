"""Summary of generated traces.

Answers two questions: how many usable clean traces exist, and what the agent's
natural success rate was. The second is a reported number on the poster, so it
is computed from the traces rather than remembered from a console log.

Also cross-tabulates success against narration, since a trace that reaches the
right answer without stating any reasoning is both a likely fluke and useless
as an injection target.

    python agent/trace_stats.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "env"))

import yaml  # noqa: E402


def load(dirpath: Path) -> list[dict]:
    return [json.loads(f.read_text()) for f in sorted(dirpath.glob("*.json"))]


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config" / "experiment.yaml").read_text())
    clean = load(ROOT / cfg["paths"]["traces_clean"])
    failed = load(ROOT / cfg["paths"]["traces_failed"])
    traces = clean + failed
    if not traces:
        print("no traces yet")
        return 0

    n = len(traces)
    print(f"traces: {n}   clean: {len(clean)}   failed: {len(failed)}"
          f"   ({len(clean) / n:.0%} clean)")

    first_try = [t for t in traces if t["meta"].get("attempts_made", 1) == 1]
    if first_try:
        ok = sum(t["meta"]["outcome_correct"] for t in first_try)
        print(f"first-attempt success: {ok}/{len(first_try)} ({ok / len(first_try):.0%})")

    # Does narration predict success? This drove the agent-model decision.
    print("\nnarration vs outcome")
    buckets = defaultdict(lambda: [0, 0])
    for t in traces:
        m = t["meta"]
        key = "narrated" if m.get("narration_chars", 0) >= 200 else "silent"
        buckets[key][0 if m["outcome_correct"] else 1] += 1
    print(f"  {'':<10}{'correct':>9}{'wrong':>8}")
    for key in ("narrated", "silent"):
        ok, bad = buckets[key]
        if ok + bad:
            print(f"  {key:<10}{ok:>9}{bad:>8}   ({ok / (ok + bad):.0%} correct)")

    print("\nstop reasons: ", dict(Counter(t["meta"]["stop_reason"].split(":")[0]
                                           for t in traces)))
    reminded = sum(1 for t in traces if t["meta"].get("submit_reminders", 0) > 0)
    print(f"needed a submit reminder: {reminded}/{n}")

    print("\nby binding rule")
    by_rule = defaultdict(lambda: [0, 0])
    for t in traces:
        m = t["meta"]
        key = "+".join(m["binding_rules"]) or "none"
        by_rule[key][0 if m["outcome_correct"] else 1] += 1
    for key in sorted(by_rule):
        ok, bad = by_rule[key]
        print(f"  {key:<16}{ok:>3} clean /{ok + bad:>3} run")

    times = [t["meta"]["wall_time_s"] for t in traces]
    print(f"\nepisode wall time: median {sorted(times)[len(times) // 2]:.0f}s  "
          f"total {sum(times) / 60:.0f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
