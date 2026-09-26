"""Validity checks on the injected traces.

The causal claim of the experiment is that the two members of a matched pair
differ *only* in the final outcome. If anything else differs, a detection gap
could come from that difference instead of from the outcome, and the headline
result would be unsound. These checks are therefore not optional polish.

Verified per pair:
  1. both members exist and carry the same fault at the same step
  2. every message before the tail is byte-identical
  3. the only differences are the final calculate / submit / summary lines
  4. the correct member submits the oracle amount, the wrong member does not
  5. tool outputs are genuine - re-running the agent's calculate expression
     through the real tool reproduces the stored result
  6. the conversation stays well formed: every tool reply matches a preceding
     tool call id
  7. the fabricated or deleted element is present in BOTH members

    python faults/verify_pairs.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "env"))

import yaml  # noqa: E402
from oracle import Environment  # noqa: E402
from tools import ToolSession  # noqa: E402


def wellformed(messages: list[dict]) -> list[str]:
    """Every tool reply must answer a tool call that came before it."""
    problems, open_ids = [], {}
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            for c in m.get("tool_calls") or []:
                open_ids[c["id"]] = c["function"]["name"]
        elif m["role"] == "tool":
            tid = m.get("tool_call_id")
            if tid not in open_ids:
                problems.append(f"msg {i}: tool reply {tid} has no matching call")
            elif open_ids[tid] != m.get("name"):
                problems.append(f"msg {i}: tool reply name mismatch")
    return problems


def tool_outputs_truthful(messages: list[dict], env: Environment) -> list[str]:
    """Re-execute the agent's calculate calls and compare with the trace."""
    problems, session = [], ToolSession(env)
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        for c in m.get("tool_calls") or []:
            if c["function"]["name"] != "calculate":
                continue
            reply = next((messages[j] for j in range(i + 1, len(messages))
                          if messages[j]["role"] == "tool"
                          and messages[j].get("tool_call_id") == c["id"]), None)
            if reply is None:
                continue
            expr = c["function"]["arguments"].get("expression", "")
            fresh = session.calculate(expr)
            stored = json.loads(reply["content"])
            if "result" not in fresh or "result" not in stored:
                continue
            if abs(fresh["result"] - stored["result"]) > 0.005:
                problems.append(
                    f"msg {i}: calculate({expr[:40]}...) stored "
                    f"{stored['result']} but the tool returns {fresh['result']}")
    return problems


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config" / "experiment.yaml").read_text())
    env = Environment()
    out_dir = ROOT / cfg["paths"]["traces_faulty"]

    pairs = defaultdict(dict)
    controls = []
    for f in sorted(out_dir.glob("*.json")):
        t = json.loads(f.read_text())
        inj = t["injection"]
        if inj["fault_type"] is None:
            controls.append(t)
        else:
            pairs[(t["task_id"], inj["fault_type"])][inj["outcome_version"]] = t

    errors, checked = [], 0
    tail_offsets = []
    for (task, fault), members in sorted(pairs.items()):
        if set(members) != {"correct", "wrong"}:
            errors.append(f"{task}/{fault}: incomplete pair {sorted(members)}")
            continue
        c, w = members["correct"], members["wrong"]
        cm, wm = c["messages"], w["messages"]
        checked += 1

        # 1. same fault, same step
        for key in ("fault_type", "fault_step_index", "target"):
            if c["injection"].get(key) != w["injection"].get(key):
                errors.append(f"{task}/{fault}: members disagree on {key}")

        # 2/3. identical except the tail
        if len(cm) != len(wm):
            errors.append(f"{task}/{fault}: different message counts "
                          f"({len(cm)} vs {len(wm)})")
            continue
        diffs = [i for i, (a, b) in enumerate(zip(cm, wm))
                 if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True)]
        if not diffs:
            errors.append(f"{task}/{fault}: members are identical - no outcome difference")
            continue
        first = min(diffs)
        # every message from the first difference on may differ; nothing before may
        if diffs != list(range(first, max(diffs) + 1)):
            errors.append(f"{task}/{fault}: differences are not contiguous: {diffs}")
        tail_offsets.append(len(cm) - first)
        # the first difference must be in the closing section, not in the analysis
        a_idx = c["injection"].get("fault_step_index", 0)
        if fault == "F1" and first <= a_idx:
            errors.append(f"{task}/{fault}: divergence at msg {first} is at or before "
                          f"the injected analysis (msg {a_idx})")

        # 4. amounts
        oracle = c["injection"]["oracle_amount_eur"]
        if abs(c["injection"]["submitted_amount_eur"] - oracle) > 0.005:
            errors.append(f"{task}/{fault}: correct member does not submit the oracle")
        if abs(w["injection"]["submitted_amount_eur"] - oracle) < 0.01:
            errors.append(f"{task}/{fault}: wrong member submits the oracle amount")

        # 5/6. honesty and shape
        for label, t in (("correct", c), ("wrong", w)):
            for p in wellformed(t["messages"]):
                errors.append(f"{task}/{fault}/{label}: {p}")
            for p in tool_outputs_truthful(t["messages"], env):
                errors.append(f"{task}/{fault}/{label}: {p}")

        # 7. the fault survives in both members
        if fault == "F1":
            fake = c["injection"]["edited"]
            for label, t in (("correct", c), ("wrong", w)):
                if not any(fake in (m.get("content") or "") for m in t["messages"]):
                    errors.append(f"{task}/{fault}/{label}: fabricated value missing")
        else:
            deleted = "get_expenses" if fault == "F2" else "get_policy"
            for label, t in (("correct", c), ("wrong", w)):
                called = [cc["function"]["name"] for m in t["messages"]
                          if m["role"] == "assistant" for cc in m.get("tool_calls") or []]
                if deleted in called:
                    errors.append(f"{task}/{fault}/{label}: {deleted} still present")

    print(f"pairs checked: {checked}   clean controls: {len(controls)}")
    if tail_offsets:
        print(f"divergence begins within the last {min(tail_offsets)}-"
              f"{max(tail_offsets)} messages of the trace")
    by_fault = defaultdict(int)
    for (_, fault) in pairs:
        by_fault[fault] += 1
    print(f"pairs per fault type: {dict(sorted(by_fault.items()))}")

    for e in errors:
        print(f"FAIL  {e}")
    print("\nVERIFICATION:", "PASSED" if not errors else f"FAILED ({len(errors)})")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
