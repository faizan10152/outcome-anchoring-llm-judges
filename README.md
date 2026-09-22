# Outcome Anchoring in LLM Judges of Agent Traces

Do LLM judges of agent traces detect process faults independently of whether the
final outcome is correct — or are they **outcome-anchored**?

This repository contains the full experiment behind the poster for *Trends in NLP*
(University of Trier, Simon Münker). Everything needed to reproduce the results is
here: the environment, the agent, the fault injector, the judge runs and the analysis.

## The design in one paragraph

An agent solves employee expense-reimbursement tasks using deterministic tools.
Traces that reach the correct total are kept as *clean* traces. Into each clean
trace we inject exactly one **process fault** (a fabricated observation, a phantom
action, or a skipped check). Each faulty trace then exists as a **matched pair**
that is byte-identical except for the final lines: in one version the agent submits
the correct amount, in the other a wrong one. LLM judges see the full trace and say
whether the agent handled the task correctly. Because the two members of a pair
contain the *same* fault, any difference in detection is attributable to the outcome
alone. The headline metric is the **anchoring gap**:

```
anchoring_gap = detection_rate(outcome wrong) − detection_rate(outcome correct)
```

### Hypotheses

| | |
|---|---|
| **H1** (anchoring) | Fault detection is substantially lower when the final outcome is correct. |
| **H2** (fault type) | Fabricated observations and phantom actions are detected less often than skipped checks. |
| **H3** (reasoning)  | Reason-then-verdict reduces the anchoring gap but does not eliminate it. |

A null result is reportable: no gap would mean judges miss faults uniformly rather
than because of the outcome.

## The environment

Employee expense reimbursement was chosen because the outcome is a single number
(the reimbursed amount), so *correct* and *wrong* are unambiguous, and because every
policy rule has a checkable consequence in the trace.

The policy (`env/data/policy.json`) has four rules, applied in this order:

| Rule | Text |
|---|---|
| **R1** | Alcohol is never reimbursable; the alcohol portion is deducted. |
| **R2** | Taxi expenses are reimbursed only with a receipt. |
| **R3** | Any expense charged above €25.00 requires a receipt (tested on the charged amount, before R1). |
| **R4** | Eligible meal spend is capped at €40.00 per calendar day (after R1–R3). |

The agent has six tools: `list_employees`, `get_employee`, `get_policy`,
`get_expenses`, `calculate`, `submit_reimbursement`.

**36 tasks** over 12 employees and 36 trips (7–11 expense items each). Each trip is
generated from a scenario that controls exactly which rules *bind* — i.e. which rules
actually change the reimbursed total. This matters: a rule that does not bind cannot
be expressed as a wrong final amount, so it cannot carry a matched pair. `env/validate.py`
verifies the intended bindings hold for every trip.

| Binding rule | Trips |
|---|---|
| R1 (alcohol) | 12 |
| R2 (taxi receipt) | 12 |
| R3 (receipt threshold) | 12 |
| R4 (meal cap) | 13 |
| none (clean scenario) | 4 |

The **oracle** (`env/oracle.py`) computes the true reimbursement independently of the
agent, in whole cents so results are exact. It also computes *counterfactual* amounts
— under a perturbed policy (`policy_override`) or with a rule ignored
(`disabled_rules`) — which is how the fault injector derives the wrong amount that
each fault itself implies, rather than an arbitrary wrong number.

## Reproducing

```bash
git clone https://github.com/faizan10152/outcome-anchoring-llm-judges.git
cd outcome-anchoring-llm-judges

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python env/generate_data.py     # regenerate the database (deterministic, seeded)
python env/validate.py          # integrity + rule-binding checks
python env/tasks.py             # write data/tasks.json
pytest -q                       # oracle and tool test suite
```

The generated database and all traces are committed, so the judge runs and the
analysis can be reproduced without re-running the agent.

## Layout

```
config/experiment.yaml   models, seeds, conditions — the run definition
env/                     database, policy, tools, oracle, tasks, validation
agent/run_agent.py       ReAct-style agent loop -> data/traces/clean|failed
faults/inject.py         F1-F3 injection + matched outcome pairs
judge/                   judge prompts (direct, reason-then-verdict) and runner
analysis/                detection rates, anchoring gap, statistics, figures
data/                    tasks and generated traces (committed)
results/                 judgments and poster figures
tests/                   hand-computed oracle checks, tool checks
```

## Status

- [x] Environment: database, policy, tools, oracle, 36 tasks, validation, tests
- [ ] Agent loop and clean-trace generation
- [ ] Fault injection and matched outcome pairs
- [ ] Judge runs
- [ ] Analysis and figures

## Open decisions

- ~~Third judge model.~~ Resolved: the lineup is `qwen2.5:7b`, `llama3.1:8b` and
  `qwen3:14b` — three families across two size bands, run locally via ollama.
- **Agent model.** `qwen3:14b` is the default; it has to yield roughly 30 of 36
  clean traces to be viable. Decided empirically on the first full agent run.
- **Matched-pair construction.** Default is *option A*: the pair differs only in the
  final lines. In the outcome-correct version the faulty step can look inconsistent
  with the correct total, which is an extra cue and makes detection *easier* in that
  condition — i.e. it biases against H1 and makes the test conservative. This is
  accepted and reported as a limitation.

## Licence

Released under the MIT Licence — see [LICENSE](LICENSE). The code, the generated
environment and the traces may be reused freely, including to replicate or extend
these experiments.

The employee names, trips and expense items in `env/data/` are synthetic, produced
by `env/generate_data.py` from a fixed seed. They describe no real person.
