"""Supplement upstream partial-state matching with exact reservation checks."""

from __future__ import annotations

from shared.grading.matching import expected_arguments_match


def audit_state(scenario: dict, detail: dict) -> dict:
    task = detail.get("task_metrics", {})
    final = detail.get("final_state", task.get("final_application_state"))
    initial = scenario["application"]["initial_state"]
    expected = scenario["expected"]["state"]
    if not isinstance(final, dict):
        return {"passed": False, "reason": "Final application state is missing", "evidence_error": True}
    if expected.get("unchanged"):
        passed = final == initial
    else:
        wanted = expected.get("reservations", [])
        actual = final.get("reservations", [])
        passed = len(actual) == len(wanted) and expected_arguments_match(expected, final)
    return {
        "passed": passed,
        "reason": "Exact reservation count and expected state checked",
        "final_state": final,
        "expected_state": expected,
        "evidence_error": False,
    }
