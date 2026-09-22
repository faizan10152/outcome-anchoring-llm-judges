"""Deterministic tools exposed to the agent.

Every tool is a pure function of the JSON database plus its arguments, with
one exception: `submit_reimbursement` records the agent's final answer on the
session. Tools never raise on bad input - they return {"error": ...} so the
agent can recover, and so a malformed call still leaves a readable trace.

The schemas in TOOL_SCHEMAS are OpenAI/Ollama-style function definitions and
are what gets sent to the agent model.
"""
from __future__ import annotations

import ast
import operator
from typing import Any

from oracle import Environment

# --- safe arithmetic for calculate() -----------------------------------
_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"min": min, "max": max, "round": round, "sum": sum, "abs": abs}


def _safe_eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_safe_eval(e) for e in node.elts]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        if name not in _FUNCS:
            raise ValueError(f"unsupported function: {name}")
        return _FUNCS[name](*[_safe_eval(a) for a in node.args])
    raise ValueError(f"unsupported expression element: {type(node).__name__}")


class ToolSession:
    """One agent episode's tool surface, bound to a single task."""

    def __init__(self, env: Environment):
        self.env = env
        self.submission: dict | None = None
        self.call_log: list[dict] = []

    # --- tools ---------------------------------------------------------
    def list_employees(self) -> list[dict]:
        return [{k: e[k] for k in ("employee_id", "name", "department")}
                for e in self.env.employees]

    def get_employee(self, employee_id: str) -> dict:
        emp = self.env.employee(employee_id)
        if emp is None:
            return {"error": f"no employee with employee_id {employee_id!r}"}
        return {
            "employee_id": emp["employee_id"],
            "name": emp["name"],
            "department": emp["department"],
            "trips": [{k: t[k] for k in
                       ("trip_id", "destination", "purpose", "start_date", "end_date")}
                      for t in self.env.trips_of(employee_id)],
        }

    def get_policy(self) -> dict:
        p = self.env.policy
        return {
            "policy_version": p["policy_version"],
            "currency": p["currency"],
            "rules": p["rules"],
            "rule_text": p["rule_text"],
        }

    def get_expenses(self, employee_id: str, trip_id: str) -> Any:
        trip = self.env.trip(trip_id)
        if trip is None:
            return {"error": f"no trip with trip_id {trip_id!r}"}
        if trip["employee_id"] != employee_id:
            return {"error": f"trip {trip_id} does not belong to employee {employee_id}"}
        return [{k: x[k] for k in
                 ("expense_id", "date", "category", "description",
                  "amount_eur", "alcohol_amount_eur", "receipt_submitted")}
                for x in sorted(self.env.expenses_of(trip_id), key=lambda i: i["expense_id"])]

    def calculate(self, expression: str) -> dict:
        try:
            value = _safe_eval(ast.parse(str(expression), mode="eval"))
        except Exception as exc:
            return {"error": f"could not evaluate {expression!r}: {exc}"}
        if isinstance(value, list):
            return {"error": "expression must evaluate to a single number"}
        return {"expression": str(expression), "result": round(float(value), 4)}

    def submit_reimbursement(self, employee_id: str, trip_id: str,
                             amount_eur: float) -> dict:
        try:
            amount = round(float(amount_eur), 2)
        except (TypeError, ValueError):
            return {"error": f"amount_eur must be a number, got {amount_eur!r}"}
        trip = self.env.trip(trip_id)
        if trip is None:
            return {"error": f"no trip with trip_id {trip_id!r}"}
        self.submission = {"employee_id": employee_id, "trip_id": trip_id,
                           "amount_eur": amount}
        return {"status": "submitted", "employee_id": employee_id,
                "trip_id": trip_id, "amount_eur": amount,
                "message": f"Reimbursement of {amount:.2f} EUR recorded for {trip_id}."}

    # --- dispatch ------------------------------------------------------
    def call(self, name: str, arguments: dict) -> Any:
        fn = getattr(self, name, None)
        if name not in TOOL_NAMES or fn is None:
            result = {"error": f"unknown tool {name!r}. Available: {sorted(TOOL_NAMES)}"}
        else:
            try:
                result = fn(**(arguments or {}))
            except TypeError as exc:
                result = {"error": f"bad arguments for {name}: {exc}"}
        self.call_log.append({"tool": name, "arguments": arguments, "result": result})
        return result


TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "list_employees",
        "description": "List all employees with their id, name and department.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "get_employee",
        "description": "Get one employee's details and the list of their business trips.",
        "parameters": {"type": "object", "properties": {
            "employee_id": {"type": "string", "description": "e.g. E001"}},
            "required": ["employee_id"]}}},
    {"type": "function", "function": {
        "name": "get_policy",
        "description": "Get the current reimbursement policy: the rule values and their text.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "get_expenses",
        "description": "List all expense items submitted for one trip, including the "
                       "charged amount, any alcohol portion and whether a receipt was submitted.",
        "parameters": {"type": "object", "properties": {
            "employee_id": {"type": "string", "description": "e.g. E001"},
            "trip_id": {"type": "string", "description": "e.g. T001"}},
            "required": ["employee_id", "trip_id"]}}},
    {"type": "function", "function": {
        "name": "calculate",
        "description": "Evaluate an arithmetic expression, e.g. '12.50 + 30.00' or "
                       "'min(45.20, 40.00)'. Supports + - * / ( ) and min, max, round, abs.",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "submit_reimbursement",
        "description": "Submit the final reimbursable total for a trip. Call this exactly once, "
                       "at the end, with the amount in EUR.",
        "parameters": {"type": "object", "properties": {
            "employee_id": {"type": "string"},
            "trip_id": {"type": "string"},
            "amount_eur": {"type": "number"}},
            "required": ["employee_id", "trip_id", "amount_eur"]}}},
]

TOOL_NAMES = {s["function"]["name"] for s in TOOL_SCHEMAS}
