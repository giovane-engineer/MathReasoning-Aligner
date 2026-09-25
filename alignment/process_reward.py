"""Step-level process reward evaluation for symbolic chain-of-thought."""

import re
from typing import Dict, List, Optional, Sequence

import sympy as sp

from data.validator import verify_symbolic_equivalence


def _clean_step(step: str) -> str:
    cleaned = re.sub(r"^\s*(?:step\s*)?\d+\s*[:.)-]\s*", "", step, flags=re.IGNORECASE)
    math_block = re.search(r"\$(.+?)\$", cleaned)
    return (math_block.group(1) if math_block else cleaned).strip()


def _residual(step: str):
    cleaned = _clean_step(step).replace("^", "**")
    if "=" in cleaned:
        left, right = cleaned.split("=", 1)
        return sp.sympify(left.strip()) - sp.sympify(right.strip())
    return sp.sympify(cleaned)


def _suggest_correction(previous: str, current: str) -> str:
    combined = f"{previous} {current}".lower()
    if "/" in combined or "divide" in combined:
        return (
            "Recompute the division without cancelling a potentially zero term; "
            "branch on the denominator being zero and non-zero."
        )
    if "log" in combined or "ln" in combined:
        return "Add the real-domain restriction argument > 0 before applying logarithm rules."
    if "sqrt" in combined or "root" in combined or "±" in combined:
        return "Recompute the polynomial roots and preserve every valid branch, including ± roots."
    return "Re-derive this transition and verify that both sides have the same symbolic residual."


def _transition_is_valid(previous: str, current: str) -> bool:
    try:
        previous_residual = _residual(previous)
        current_residual = _residual(current)
    except (TypeError, SyntaxError, ValueError, sp.SympifyError):
        return False
    return verify_symbolic_equivalence(str(previous_residual), str(current_residual))


def verify_cot_steps(steps: Sequence[str]) -> Dict[str, object]:
    """Score each CoT transition and report the first invalid step.

    ``error_index`` is zero-based and points to the current step in the first
    invalid transition. The returned ``self_correction`` is suitable for
    appending as a corrective training step.
    """
    if not steps:
        return {
            "valid": False,
            "step_scores": [],
            "error_index": 0,
            "self_correction": "Provide at least one symbolic reasoning step.",
        }

    step_scores: List[float] = [1.0]
    error_index: Optional[int] = None
    self_correction: Optional[str] = None

    for index in range(1, len(steps)):
        valid_transition = _transition_is_valid(steps[index - 1], steps[index])
        step_scores.append(1.0 if valid_transition else 0.0)
        if not valid_transition and error_index is None:
            error_index = index
            self_correction = _suggest_correction(steps[index - 1], steps[index])

    return {
        "valid": error_index is None,
        "step_scores": step_scores,
        "error_index": error_index,
        "self_correction": self_correction,
    }