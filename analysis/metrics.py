"""Detection rates, the anchoring gap, and the paired statistics.

The headline metric is the anchoring gap, per judge and readout:

    gap = detection(outcome wrong) - detection(outcome correct)

Detection means the judge said the agent did NOT handle the task correctly.
Because the two members of a pair contain the same fault at the same step and
differ only in the final outcome, a positive gap means the judge's willingness
to report a process fault depends on whether the answer came out right.

The pairing is exploited in the statistics: McNemar's test on the discordant
pairs, and a bootstrap over pairs for the confidence interval. Resampling whole
pairs (not individual judgements) keeps the dependence structure intact.

    python analysis/metrics.py
    python analysis/metrics.py --csv results/cells.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "env"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from scipy import stats as sps  # noqa: E402
from statsmodels.stats.contingency_tables import mcnemar  # noqa: E402


def load_judgments(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def rate(values: list[bool]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def bootstrap_gap_ci(pairs: list[tuple[bool, bool]], resamples: int = 10000,
                     level: float = 0.95, rng_seed: int = 20260922):
    """CI for the gap, resampling whole pairs."""
    if not pairs:
        return (None, None)
    rng = np.random.default_rng(rng_seed)
    arr = np.array(pairs, dtype=float)          # columns: (wrong, correct)
    idx = rng.integers(0, len(arr), size=(resamples, len(arr)))
    gaps = arr[idx, 0].mean(axis=1) - arr[idx, 1].mean(axis=1)
    lo = float(np.quantile(gaps, (1 - level) / 2))
    hi = float(np.quantile(gaps, 1 - (1 - level) / 2))
    return lo, hi


def mcnemar_on_pairs(pairs: list[tuple[bool, bool]]):
    """Exact McNemar on (detected_wrong, detected_correct) pairs."""
    b = sum(1 for w, c in pairs if w and not c)   # only the wrong member caught
    c_ = sum(1 for w, c in pairs if c and not w)  # only the correct member caught
    if b + c_ == 0:
        return b, c_, None
    table = [[sum(1 for w, c in pairs if w and c), b],
             [c_, sum(1 for w, c in pairs if not w and not c)]]
    res = mcnemar(table, exact=True)
    return b, c_, float(res.pvalue)


def build_pairs(rows: list[dict]):
    """Group judgements into matched pairs keyed by judge, readout, trace, fault."""
    by_key = defaultdict(dict)
    for r in rows:
        if r["fault_type"] is None or r["detected"] is None:
            continue
        key = (r["judge_id"], r["readout"], r["task_id"], r["fault_type"])
        by_key[key][r["outcome_version"]] = r["detected"]
    pairs = {}
    for key, members in by_key.items():
        if {"correct", "wrong"} <= set(members):
            pairs[key] = (members["wrong"], members["correct"])
    return pairs


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config" / "experiment.yaml").read_text())
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judgments", default=cfg["paths"]["judgments"])
    ap.add_argument("--csv", help="also write the per-cell table here")
    args = ap.parse_args()

    path = ROOT / args.judgments
    if not path.exists():
        print(f"no judgments at {path}", file=sys.stderr)
        return 2
    rows = load_judgments(path)
    print(f"judgements: {len(rows)}")

    unparsed = [r for r in rows if r["verdict"] is None]
    dirty = [r for r in rows if r["verdict"] is not None
             and not r.get("verdict_parsed_cleanly", True)]
    errs = [r for r in rows if r.get("error")]
    print(f"unparseable: {len(unparsed)} ({len(unparsed) / max(1,len(rows)):.1%})   "
          f"recovered by fallback: {len(dirty)}   backend errors: {len(errs)}")

    # ---- false positives on clean traces --------------------------------
    print("\nfalse-positive rate on clean traces (judge says No when nothing is wrong)")
    print(f"  {'judge':<14}{'readout':<22}{'FPR':>7}{'n':>5}")
    for judge in sorted({r["judge_id"] for r in rows}):
        for readout in sorted({r["readout"] for r in rows}):
            clean = [r["detected"] for r in rows
                     if r["fault_type"] is None and r["judge_id"] == judge
                     and r["readout"] == readout]
            fpr = rate(clean)
            if fpr is not None:
                print(f"  {judge:<14}{readout:<22}{fpr:>6.0%}{len(clean):>5}")

    # ---- detection by cell ----------------------------------------------
    print("\ndetection rate by judge x readout x fault x outcome")
    print(f"  {'judge':<14}{'readout':<22}{'fault':<6}"
          f"{'correct':>9}{'wrong':>8}{'gap':>8}")
    cells = []
    for judge in sorted({r["judge_id"] for r in rows}):
        for readout in sorted({r["readout"] for r in rows}):
            for fault in sorted({r["fault_type"] for r in rows if r["fault_type"]}):
                sub = [r for r in rows if r["judge_id"] == judge
                       and r["readout"] == readout and r["fault_type"] == fault]
                dc = rate([r["detected"] for r in sub if r["outcome_version"] == "correct"])
                dw = rate([r["detected"] for r in sub if r["outcome_version"] == "wrong"])
                if dc is None or dw is None:
                    continue
                gap = dw - dc
                cells.append(dict(judge=judge, readout=readout, fault=fault,
                                  detection_correct=dc, detection_wrong=dw, gap=gap,
                                  n_pairs=len(sub) // 2))
                print(f"  {judge:<14}{readout:<22}{fault:<6}"
                      f"{dc:>8.0%}{dw:>8.0%}{gap:>+8.0%}")

    # ---- the headline: anchoring gap with statistics ---------------------
    pairs = build_pairs(rows)
    print("\nANCHORING GAP (pooled over fault types), with McNemar and bootstrap CI")
    print(f"  {'judge':<14}{'readout':<22}{'gap':>7}{'95% CI':>18}"
          f"{'b':>5}{'c':>5}{'p':>9}{'pairs':>7}")
    for judge in sorted({k[0] for k in pairs}):
        for readout in sorted({k[1] for k in pairs}):
            sel = [v for k, v in pairs.items() if k[0] == judge and k[1] == readout]
            if not sel:
                continue
            gap = rate([w for w, _ in sel]) - rate([c for _, c in sel])
            lo, hi = bootstrap_gap_ci(sel, cfg["analysis"]["bootstrap_resamples"],
                                      cfg["analysis"]["ci_level"])
            b, c_, p = mcnemar_on_pairs(sel)
            ci = f"[{lo:+.0%}, {hi:+.0%}]" if lo is not None else "n/a"
            pv = f"{p:.4f}" if p is not None else "n/a"
            print(f"  {judge:<14}{readout:<22}{gap:>+7.0%}{ci:>18}"
                  f"{b:>5}{c_:>5}{pv:>9}{len(sel):>7}")

    # ---- H3: does reasoning shrink the gap? -----------------------------
    print("\nH3: gap by readout (pooled over judges and fault types)")
    for readout in sorted({k[1] for k in pairs}):
        sel = [v for k, v in pairs.items() if k[1] == readout]
        if sel:
            gap = rate([w for w, _ in sel]) - rate([c for _, c in sel])
            lo, hi = bootstrap_gap_ci(sel)
            print(f"  {readout:<22}gap {gap:+.0%}  95% CI [{lo:+.0%}, {hi:+.0%}]  "
                  f"({len(sel)} pairs)")

    # ---- H2: fault-type ordering ---------------------------------------
    print("\nH2: detection by fault type (pooled over judges and readouts)")
    for fault in sorted({k[3] for k in pairs}):
        sel = [v for k, v in pairs.items() if k[3] == fault]
        dw, dc = rate([w for w, _ in sel]), rate([c for _, c in sel])
        print(f"  {fault:<6}correct {dc:>5.0%}   wrong {dw:>5.0%}   "
              f"gap {dw - dc:+.0%}   ({len(sel)} pairs)")

    # ---- localisation ---------------------------------------------------
    loc = [r for r in rows if r["readout"] != "direct" and r["fault_type"]
           and r["detected"] and r.get("localized_step") is not None]
    if loc:
        hits = sum(1 for r in loc
                   if r["fault_step_index"] is not None
                   and abs(r["localized_step"] - r["fault_step_index"]) <= 1)
        print(f"\nstep localisation: {hits}/{len(loc)} within +-1 of the injected "
              f"step ({hits / len(loc):.0%}) among detected faults in the "
              f"reasoning condition")

    if args.csv:
        import csv
        out = ROOT / args.csv
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(cells[0]))
            w.writeheader()
            w.writerows(cells)
        print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
