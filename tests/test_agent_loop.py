"""Agent-loop control flow, exercised with a scripted stub instead of a model.

The loop decides what counts as a clean trace, so its bookkeeping is tested
directly: a model that submits the oracle amount must be recorded as correct,
and every other ending must be recorded as a failure with a reason.
"""
import json

import pytest
from llm import LLMError, normalise_tool_calls
from oracle import Environment, oracle_amount
from run_agent import run_episode
from tasks import load_tasks


class ScriptedClient:
    """Replays a fixed list of assistant messages, one per chat() call."""

    def __init__(self, replies, model="stub:test"):
        self.replies, self.calls, self.model = list(replies), 0, model

    def settings(self):
        return {"backend": "stub", "model": self.model}

    def chat(self, messages, tools=None):
        if self.calls >= len(self.replies):
            raise LLMError("script exhausted")
        reply = self.replies[self.calls]
        self.calls += 1
        return reply


def tool_call(name, args):
    return {"function": {"name": name, "arguments": args}}


@pytest.fixture
def env():
    return Environment()


@pytest.fixture
def task():
    return next(t for t in load_tasks() if t["task_id"] == "K001")


def test_correct_submission_is_marked_clean(env, task):
    truth = oracle_amount(env, task["trip_id"])
    client = ScriptedClient([
        {"content": "Let me check the policy.",
         "tool_calls": [tool_call("get_policy", {})]},
        {"content": "Submitting the total.",
         "tool_calls": [tool_call("submit_reimbursement", {
             "employee_id": task["employee_id"], "trip_id": task["trip_id"],
             "amount_eur": truth})]},
        {"content": f"Reimbursed {truth:.2f} EUR."},
    ])
    trace = run_episode(task, env, client, max_steps=10)
    m = trace["meta"]
    assert m["outcome_correct"] is True
    assert m["submitted_amount_eur"] == truth
    assert m["stop_reason"] == "final_message"


def test_wrong_submission_is_marked_failed(env, task):
    truth = oracle_amount(env, task["trip_id"])
    client = ScriptedClient([
        {"content": "", "tool_calls": [tool_call("submit_reimbursement", {
            "employee_id": task["employee_id"], "trip_id": task["trip_id"],
            "amount_eur": round(truth + 10.0, 2)})]},
        {"content": "Done."},
    ])
    assert run_episode(task, env, client, max_steps=10)["meta"]["outcome_correct"] is False


def test_finishing_without_submitting_is_reported(env, task):
    # Two replies: the first ends the episode, the second ignores the reminder.
    client = ScriptedClient([{"content": "I think it is about 300 EUR."},
                             {"content": "As I said, about 300 EUR."}])
    m = run_episode(task, env, client, max_steps=10)["meta"]
    assert m["stop_reason"] == "finished_without_submitting"
    assert m["submitted_amount_eur"] is None
    assert m["outcome_correct"] is False


def test_max_steps_stops_a_looping_agent(env, task):
    client = ScriptedClient([{"content": "checking again",
                              "tool_calls": [tool_call("get_policy", {})]}] * 20)
    m = run_episode(task, env, client, max_steps=4)["meta"]
    assert m["stop_reason"] == "max_steps"
    assert m["assistant_turns"] == 4


def test_llm_error_is_captured_not_raised(env, task):
    m = run_episode(task, env, ScriptedClient([]), max_steps=5)["meta"]
    assert m["stop_reason"].startswith("llm_error")
    assert m["outcome_correct"] is False


def test_every_tool_call_gets_a_matching_tool_message(env, task):
    client = ScriptedClient([
        {"content": "two lookups",
         "tool_calls": [tool_call("get_policy", {}),
                        tool_call("get_employee", {"employee_id": task["employee_id"]})]},
        {"content": "done"},
    ])
    msgs = run_episode(task, env, client, max_steps=10)["messages"]
    call_ids = [c["id"] for m in msgs if m["role"] == "assistant"
                for c in m.get("tool_calls", [])]
    tool_ids = [m["tool_call_id"] for m in msgs if m["role"] == "tool"]
    assert call_ids == tool_ids and len(call_ids) == 2


def test_tool_messages_carry_real_tool_output(env, task):
    client = ScriptedClient([
        {"content": "policy", "tool_calls": [tool_call("get_policy", {})]},
        {"content": "done"},
    ])
    msgs = run_episode(task, env, client, max_steps=10)["messages"]
    payload = json.loads(next(m for m in msgs if m["role"] == "tool")["content"])
    assert payload["rules"]["meal_daily_cap_eur"] == 40.0


def test_trace_hides_nothing_the_judge_needs_and_keeps_answers_in_meta(env, task):
    """The conversation must not contain the oracle amount or task metadata."""
    truth = oracle_amount(env, task["trip_id"])
    client = ScriptedClient([{"content": "no tools"}])
    trace = run_episode(task, env, client, max_steps=5)
    body = json.dumps(trace["messages"])
    assert "oracle" not in body and "binding_rules" not in body
    assert trace["meta"]["oracle_amount_eur"] == truth


def test_unknown_tool_is_logged_and_does_not_crash(env, task):
    client = ScriptedClient([
        {"content": "", "tool_calls": [tool_call("wipe_database", {})]},
        {"content": "sorry"},
    ])
    msgs = run_episode(task, env, client, max_steps=5)["messages"]
    assert "error" in json.loads(next(m for m in msgs if m["role"] == "tool")["content"])


@pytest.mark.parametrize("raw,expected_args", [
    ({"tool_calls": [{"function": {"name": "f", "arguments": {"a": 1}}}]}, {"a": 1}),
    ({"tool_calls": [{"function": {"name": "f", "arguments": '{"a": 1}'}}]}, {"a": 1}),
    ({"tool_calls": [{"function": {"name": "f", "arguments": ""}}]}, {}),
])
def test_tool_call_shapes_are_normalised(raw, expected_args):
    assert normalise_tool_calls(raw)[0]["function"]["arguments"] == expected_args


def test_missing_call_ids_are_generated():
    raw = {"tool_calls": [{"function": {"name": "a", "arguments": {}}},
                          {"function": {"name": "b", "arguments": {}}}]}
    assert [c["id"] for c in normalise_tool_calls(raw)] == ["call_0", "call_1"]


def test_agent_that_forgets_to_submit_gets_one_reminder(env, task):
    """A missing submit call is a harness problem, so it gets one nudge."""
    truth = oracle_amount(env, task["trip_id"])
    client = ScriptedClient([
        {"content": "The total is about 270 EUR."},
        {"content": "Submitting.", "tool_calls": [tool_call(
            "submit_reimbursement", {"employee_id": task["employee_id"],
                                     "trip_id": task["trip_id"], "amount_eur": truth})]},
        {"content": "Done."},
    ])
    trace = run_episode(task, env, client, max_steps=10)
    assert trace["meta"]["outcome_correct"] is True
    assert trace["meta"]["submit_reminders"] == 1


def test_the_reminder_is_sent_at_most_once(env, task):
    """A model that ignores the nudge must not be nudged forever."""
    client = ScriptedClient([{"content": "I am done."}] * 6)
    m = run_episode(task, env, client, max_steps=10)["meta"]
    assert m["submit_reminders"] == 1
    assert m["stop_reason"] == "finished_without_submitting"


def test_no_reminder_when_the_agent_did_submit(env, task):
    truth = oracle_amount(env, task["trip_id"])
    client = ScriptedClient([
        {"content": "Submitting.", "tool_calls": [tool_call(
            "submit_reimbursement", {"employee_id": task["employee_id"],
                                     "trip_id": task["trip_id"], "amount_eur": truth})]},
        {"content": "Done."},
    ])
    assert run_episode(task, env, client, max_steps=10)["meta"]["submit_reminders"] == 0
