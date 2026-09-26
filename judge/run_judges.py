"""Run the LLM judges over every trace.

For each (trace, judge model, readout) the judge is shown the task and the full
transcript and asked whether the agent handled it correctly. A verdict of "No"
counts as detection.

Two implementation points that matter:

  - The loop is judge-outer. Iterating trace-by-trace while switching models
    would make ollama unload and reload an 11 GB model on every call.
  - Results are appended to JSONL and already-finished combinations are skipped,
    so an interrupted run resumes instead of starting over.

    python judge/run_judges.py --pilot 12      # the ceiling/floor gate first
    python judge/run_judges.py                 # the full run
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "common"))
sys.path.insert(0, str(ROOT / "judge"))

import yaml  # noqa: E402
from llm import LLMError, OllamaClient  # noqa: E402
from prompts import build_prompt  # noqa: E402

WORD = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def parse_verdict(text: str, readout: str) -> tuple[str | None, bool]:
    """Extract the verdict. Returns (verdict, parsed_cleanly)."""
    if not text:
        return None, False
    if readout == "reason_then_verdict":
        # The instruction puts the verdict last, so search from the last mention.
        hits = list(re.finditer(r"verdict\s*[:\-]?\s*", text, re.IGNORECASE))
        if hits:
            tail = text[hits[-1].end():]
            m = WORD.search(tail)
            if m:
                return m.group(1).lower(), True
        # Fall back to the last bare yes/no anywhere in the reply.
        words = WORD.findall(text)
        return (words[-1].lower(), False) if words else (None, False)
    m = WORD.search(text)
    return (m.group(1).lower(), True) if m else (None, False)


def parse_step(text: str) -> int | None:
    """The localisation line, taken from the end of the reply."""
    hits = re.findall(r"step\s*[:\-]?\s*(\d{1,3})\b", text or "", re.IGNORECASE)
    return int(hits[-1]) if hits else None


def load_done(path: Path) -> set[tuple[str, str, str]]:
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((r["trace_file"], r["judge_id"], r["readout"]))
    return done


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config" / "experiment.yaml").read_text())
    jcfg = cfg["judges"]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", type=int, metavar="N",
                    help="run only the first N traces (the ceiling/floor gate)")
    ap.add_argument("--judges", nargs="*", help="judge ids to run (default: all available)")
    ap.add_argument("--readouts", nargs="*", default=jcfg["readouts"])
    ap.add_argument("--out", default=cfg["paths"]["judgments"])
    ap.add_argument("--num-ctx", type=int, default=8192)
    args = ap.parse_args()

    faulty_dir = ROOT / cfg["paths"]["traces_faulty"]
    traces = sorted(faulty_dir.glob("*.json"))
    if args.pilot:
        # The gate needs BOTH faulty and clean traces. Detection on its own is
        # uninterpretable: a judge that says "No" to everything scores high
        # detection and is useless. The clean controls give the false-positive
        # rate, and detection minus FPR is the only meaningful read.
        faults = [p for p in traces if "-clean-" not in p.name]
        clean = [p for p in traces if "-clean-" in p.name]
        traces = faults[:args.pilot] + clean
    if not traces:
        print("no traces found - run faults/inject.py first", file=sys.stderr)
        return 2

    judges = [j for j in jcfg["models"] if j.get("available")]
    if args.judges:
        wanted = set(args.judges)
        judges = [j for j in judges if j["id"] in wanted]
    if not judges:
        print("no available judges in config", file=sys.stderr)
        return 2

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)

    total = len(traces) * len(judges) * len(args.readouts)
    print(f"traces={len(traces)}  judges={[j['id'] for j in judges]}  "
          f"readouts={args.readouts}  ->  {total} calls "
          f"({len(done)} already done)\n", flush=True)

    n, started_all = 0, time.time()
    for judge in judges:                      # outer: keeps one model resident
        client = OllamaClient(model=judge["name"], base_url=jcfg["base_url"],
                              temperature=jcfg["temperature"], seed=jcfg["seed"],
                              num_ctx=args.num_ctx, think=False)
        for readout in args.readouts:
            stats = {"yes": 0, "no": 0, "unparsed": 0}
            t0 = time.time()
            for path in traces:
                key = (path.name, judge["id"], readout)
                n += 1
                if key in done:
                    continue
                trace = json.loads(path.read_text())
                inj = trace.get("injection", {})
                messages = build_prompt(trace, readout)

                started = time.time()
                try:
                    reply = client.chat(messages)
                    raw = reply.get("content") or ""
                    err = None
                except LLMError as exc:
                    raw, err = "", str(exc)

                verdict, clean = parse_verdict(raw, readout)
                record = {
                    "trace_file": path.name,
                    "task_id": trace["task_id"],
                    "trip_id": trace["meta"]["trip_id"],
                    "fault_type": inj.get("fault_type"),
                    "outcome_version": inj.get("outcome_version"),
                    "submitted_amount_eur": inj.get("submitted_amount_eur"),
                    "oracle_amount_eur": trace["meta"]["oracle_amount_eur"],
                    "binding_rules": trace["meta"]["binding_rules"],
                    "fault_step_index": inj.get("fault_step_index"),
                    "judge_id": judge["id"],
                    "judge_model": judge["name"],
                    "judge_family": judge["family"],
                    "judge_params_b": judge["params_b"],
                    "readout": readout,
                    "verdict": verdict,
                    # detection = the judge says the agent did NOT handle it correctly
                    "detected": None if verdict is None else (verdict == "no"),
                    "verdict_parsed_cleanly": clean,
                    "localized_step": parse_step(raw) if readout != "direct" else None,
                    "raw_output": raw,
                    "error": err,
                    "latency_s": round(time.time() - started, 2),
                    "judge_settings": client.settings(),
                    "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
                with out_path.open("a") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")

                if verdict is None:
                    stats["unparsed"] += 1
                else:
                    stats[verdict] += 1
                if n % 10 == 0 or verdict is None:
                    el = time.time() - started_all
                    print(f"  [{n}/{total}] {judge['id']}/{readout} "
                          f"{path.name[:34]:<34} -> {verdict or 'UNPARSED'}  "
                          f"({record['latency_s']:.1f}s, {el / 60:.0f}m elapsed)",
                          flush=True)
            dur = (time.time() - t0) / 60
            said_no = stats["no"] / max(1, stats["yes"] + stats["no"])
            print(f"{judge['id']}/{readout}: said No on {said_no:.0%} "
                  f"(yes={stats['yes']} no={stats['no']} unparsed={stats['unparsed']}) "
                  f"in {dur:.0f} min", flush=True)

    print(f"\ndone in {(time.time() - started_all) / 60:.0f} min -> "
          f"{out_path.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
