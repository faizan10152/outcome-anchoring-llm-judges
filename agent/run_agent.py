"""ReAct-style agent loop over the expense-reimbursement environment.

Runs the agent on every task, logs the full conversation, and sorts the
episodes by whether the submitted amount matches the oracle:

    data/traces/clean/   submitted == oracle   -> the substrate for the experiment
    data/traces/failed/  anything else         -> kept as a side observation

Only clean traces are used for fault injection. A trace that is already wrong
cannot serve as the outcome-correct member of a matched pair, and its failure
would confound the injected fault.

Traces are stored as OpenAI/HF-style chat JSON (system / user / assistant with
tool_calls / tool), so the fault injector can edit them as plain data and the
judges can be shown them verbatim.

    python agent/run_agent.py                 # all tasks, skipping finished ones
    python agent/run_agent.py --limit 3       # pilot
    python agent/run_agent.py --tasks K001 K005
    python agent/run_agent.py --model qwen2.5:7b --overwrite
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "env"))
sys.path.insert(0, str(ROOT / "common"))

import yaml  # noqa: E402
from audit_clean import audit  # noqa: E402
from llm import LLMError, OllamaClient, normalise_tool_calls  # noqa: E402
from oracle import Environment, oracle_amount  # noqa: E402
from tasks import SYSTEM_PROMPT, load_tasks  # noqa: E402
from tools import TOOL_SCHEMAS, ToolSession  # noqa: E402

CENT = 0.005  # tolerance for comparing two 2-decimal amounts

# A common failure is the agent writing its closing summary without ever calling
# submit_reimbursement. That is a harness problem, not a reasoning problem, so it
# gets one neutral nudge rather than being thrown away. The reminder is recorded
# in meta and is visible in the trace, and it cannot bias the headline result:
# matched pairs are built from the *same* trace, so both members contain it.
SUBMIT_REMINDER = (
    "You have not called submit_reimbursement yet, so the task is not complete. "
    "Submit the final reimbursable amount now."
)
MAX_SUBMIT_REMINDERS = 1

# Across the pilots, whether the agent wrote out its reasoning predicted whether
# it got the answer right: every silent episode was wrong, every narrated one was
# right. Prompting for narration was not reliable, and the model's own reasoning
# channel is far too slow on this hardware (~7 tok/s for a 14B, minutes per turn).
#
# So the deliberation step is made structural instead of optional. Once the agent
# holds both the policy and the expenses, it gets exactly one turn with NO tools
# available, and has to answer in prose. That buys two things at once: the
# item-by-item reasoning that correlates with being correct, and the stated claims
# that F1/F2 injection needs as targets. It happens in every episode, so it cannot
# differ between the two members of a matched pair.
ANALYSIS_PROMPT = (
    "Before doing any arithmetic, write out your analysis. Go through the expense "
    "items one at a time. For each, state its date, category and charged amount "
    "exactly as the tools returned them, then say whether it is reimbursed in "
    "full, reduced, or excluded, and name the policy rule that decides it. "
    "Remember that rules which apply per day must be applied to each date "
    "separately. Do not call any tool in this message."
)


def load_config() -> dict:
    return yaml.safe_load((ROOT / "config" / "experiment.yaml").read_text())


def run_episode(task: dict, env: Environment, client: OllamaClient,
                max_steps: int) -> dict:
    """Run one task to completion and return the trace."""
    session = ToolSession(env)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task["prompt"]}]

    stop_reason, turns, started = None, 0, time.time()
    thinking_seen = False
    reminders = 0
    analysis_done = False

    while True:
        if turns >= max_steps:
            stop_reason = "max_steps"
            break
        # The one forced deliberation turn, as soon as the agent has the
        # policy and the expense items in hand.
        called_so_far = {c["tool"] for c in session.call_log}
        force_analysis = (not analysis_done
                          and "get_policy" in called_so_far
                          and "get_expenses" in called_so_far)
        if force_analysis:
            messages.append({"role": "user", "content": ANALYSIS_PROMPT})

        try:
            reply = client.chat(messages,
                                tools=None if force_analysis else TOOL_SCHEMAS)
        except LLMError as exc:
            stop_reason = f"llm_error: {exc}"
            break
        turns += 1

        if force_analysis:
            analysis_done = True
            if reply.get("thinking"):
                thinking_seen = True
            messages.append({"role": "assistant",
                             "content": reply.get("content") or ""})
            continue

        calls = normalise_tool_calls(reply)
        assistant = {"role": "assistant", "content": reply.get("content") or ""}
        # A separate reasoning channel is deliberately not used and not stored.
        # The agent is prompted to reason visibly in `content` instead, for two
        # reasons: a judge reviewing a trace should see the claims the agent
        # actually made, and F1/F2 injection needs those claims to exist as
        # editable text. This flag records whether a model leaked one anyway.
        if reply.get("thinking"):
            thinking_seen = True
        if calls:
            assistant["tool_calls"] = calls
        messages.append(assistant)

        if not calls:
            if session.submission is None and reminders < MAX_SUBMIT_REMINDERS:
                messages.append({"role": "user", "content": SUBMIT_REMINDER})
                reminders += 1
                continue
            stop_reason = "final_message"
            break

        for call in calls:
            name = call["function"]["name"]
            args = call["function"]["arguments"]
            result = session.call(name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "name": name,
                "content": json.dumps(result, ensure_ascii=False),
            })

    submitted = session.submission["amount_eur"] if session.submission else None
    truth = oracle_amount(env, task["trip_id"])
    correct = submitted is not None and abs(submitted - truth) < CENT
    if stop_reason == "final_message" and submitted is None:
        stop_reason = "finished_without_submitting"

    submits = [c for c in session.call_log if c["tool"] == "submit_reimbursement"]
    narrating = sum(1 for m in messages
                    if m["role"] == "assistant" and m.get("tool_calls")
                    and m["content"].strip())
    # Turn count understates narration: this model tends to put its whole
    # item-by-item review into one message. What injection actually needs is
    # enough stated text to edit, so measure the volume too.
    narration_chars = sum(len(m.get("content") or "") for m in messages
                          if m["role"] == "assistant")

    trace = {
        "trace_id": f"{task['task_id']}-{client.model.replace(':', '-')}",
        "task_id": task["task_id"],
        "task_prompt": task["prompt"],
        "messages": messages,
        "tool_call_log": session.call_log,
        # --- metadata: for analysis only, never shown to a judge ---
        "meta": {
            "employee_id": task["employee_id"],
            "trip_id": task["trip_id"],
            "scenario": task["scenario"],
            "binding_rules": task["binding_rules"],
            "oracle_amount_eur": truth,
            "submitted_amount_eur": submitted,
            "outcome_correct": correct,
            "stop_reason": stop_reason,
            "assistant_turns": turns,
            "tool_calls": len(session.call_log),
            "submit_calls": len(submits),
            "narrated_tool_turns": narrating,
            "narration_chars": narration_chars,
            "submit_reminders": reminders,
            "forced_analysis": analysis_done,
            "model_emitted_thinking": thinking_seen,
            "agent": client.settings(),
            "wall_time_s": round(time.time() - started, 2),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }
    # A correct total is not enough: the episode also has to be sound enough to
    # carry an injected fault. Record the audit verdict here so the retry loop
    # can reject a right-answer-wrong-process episode instead of keeping it.
    trace["meta"]["audit_problems"] = audit(trace)
    trace["meta"]["usable"] = correct and not trace["meta"]["audit_problems"]
    return trace


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=cfg["agent"]["model"])
    ap.add_argument("--limit", type=int, help="run only the first N tasks")
    ap.add_argument("--tasks", nargs="*", help="explicit task ids")
    ap.add_argument("--max-steps", type=int, default=cfg["agent"]["max_steps"])
    ap.add_argument("--overwrite", action="store_true",
                    help="re-run tasks whose trace already exists")
    ap.add_argument("--attempts", type=int, default=1,
                    help="retries per task, each with a different seed, until the "
                         "submitted amount matches the oracle (default 1)")
    ap.add_argument("--temperature", type=float,
                    help="override the configured decoding temperature. Seed-only "
                         "resampling is ineffective at temperature 0, so retries of "
                         "a failed task need this above 0 to explore at all")
    ap.add_argument("--only-missing", action="store_true",
                    help="run only tasks that have no usable trace yet, ignoring "
                         "any existing failed trace for them")
    ap.add_argument("--think", choices=["on", "off"], default="off",
                    help="allow a model's separate reasoning channel (default off)")
    args = ap.parse_args()

    clean_dir = ROOT / cfg["paths"]["traces_clean"]
    failed_dir = ROOT / cfg["paths"]["traces_failed"]
    clean_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)

    env = Environment()
    tasks = load_tasks()
    if args.tasks:
        wanted = set(args.tasks)
        tasks = [t for t in tasks if t["task_id"] in wanted]
    if args.limit:
        tasks = tasks[:args.limit]

    client = OllamaClient(
        model=args.model,
        base_url=cfg["agent"]["base_url"],
        temperature=(args.temperature if args.temperature is not None
                     else cfg["agent"]["temperature"]),
        seed=cfg["agent"]["seed"],
        num_ctx=cfg["agent"]["num_ctx"],
        think=False if args.think == "off" else None,
    )
    if args.model not in client.available_models():
        print(f"ERROR: model {args.model!r} not available in ollama", file=sys.stderr)
        return 2

    slug = args.model.replace(":", "-")
    print(f"agent={args.model}  tasks={len(tasks)}  max_steps={args.max_steps}\n")
    print(f"{'task':<6}{'submitted':>11}{'oracle':>10}  {'ok':<4}"
          f"{'turns':>6}{'calls':>6}{'narr':>6}{'time':>8}  stop")

    n_clean = n_failed = n_skipped = n_first = n_audit_rejects = 0
    for task in tasks:
        name = f"{task['task_id']}-{slug}.json"
        if args.only_missing:
            if (clean_dir / name).exists():
                n_skipped += 1
                continue
        elif not args.overwrite and ((clean_dir / name).exists()
                                     or (failed_dir / name).exists()):
            n_skipped += 1
            continue

        # Retry with a fresh seed until the outcome is correct. Clean traces are
        # the *substrate* of the experiment, not a measurement of agent skill, so
        # resampling a task is legitimate: what matters is that the kept trace is
        # a genuine unedited episode that reached the right answer. The cost is
        # recorded (attempts_made) and reported, because the first-attempt rate is
        # the honest number to quote for natural agent success.
        trace = None
        for attempt in range(1, args.attempts + 1):
            client.seed = cfg["agent"]["seed"] + attempt - 1
            trace = run_episode(task, env, client, args.max_steps)
            trace["meta"]["attempt_index"] = attempt
            trace["meta"]["attempts_made"] = attempt
            if trace["meta"]["usable"]:
                break
        m = trace["meta"]
        first_try = m["usable"] and m["attempts_made"] == 1
        target = clean_dir if m["usable"] else failed_dir
        # A trace can only live in one place; clear any stale copy.
        for d in (clean_dir, failed_dir):
            (d / name).unlink(missing_ok=True)
        (target / name).write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n")

        n_clean += m["usable"]
        n_failed += not m["usable"]
        n_first += first_try
        n_audit_rejects += m["outcome_correct"] and not m["usable"]
        sub = f"{m['submitted_amount_eur']:.2f}" if m["submitted_amount_eur"] is not None else "-"
        print(f"{task['task_id']:<6}{sub:>11}{m['oracle_amount_eur']:>10.2f}  "
              f"{'OK' if m['usable'] else 'FAIL':<4}{m['assistant_turns']:>6}"
              f"{m['tool_calls']:>6}{m['narrated_tool_turns']:>6}"
              f"{m['wall_time_s']:>7.1f}s  {m['stop_reason']}"
              + (f"  [{m['attempts_made']} attempts]" if m["attempts_made"] > 1 else ""))

    done = n_clean + n_failed
    if n_audit_rejects:
        print(f"\n{n_audit_rejects} episode(s) hit the right total but failed the "
              f"process audit and were rejected")
    print(f"\nclean {n_clean}/{done}" + (f"  (skipped {n_skipped} existing)" if n_skipped else ""))
    if done:
        print(f"success rate: {n_clean / done:.0%} overall"
              + (f"  |  {n_first / done:.0%} on the first attempt"
                 if args.attempts > 1 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
