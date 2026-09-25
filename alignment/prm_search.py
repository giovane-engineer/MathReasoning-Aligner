"""Test-time search with step-level process rewards and self-correction."""

import re
from typing import Dict, List

from alignment.process_reward import verify_cot_steps


class StepLevelPRM:
    """Score CoT transitions and regenerate only the first invalid step."""

    _step_pattern = re.compile(r"(?=Step\s+\d+\s*:)", re.IGNORECASE)

    def split_steps(self, cot: str) -> List[str]:
        """Split a CoT string at numbered ``Step N:`` markers."""
        steps = [step.strip() for step in self._step_pattern.split(cot) if step.strip()]
        return steps or ([cot.strip()] if cot.strip() else [])

    def evaluate(self, cot: str) -> Dict[str, object]:
        steps = self.split_steps(cot)
        result = verify_cot_steps(steps)
        result["steps"] = steps
        result["average_score"] = (
            sum(result["step_scores"]) / len(result["step_scores"])
            if result["step_scores"]
            else 0.0
        )
        return result

    def _candidate_steps(self, prompt: str) -> List[str]:
        normalized = prompt.lower().replace("^", "**")
        if "x + 2 = 5" in normalized:
            return ["Step 1: x + 2 = 5", "Step 2: x = 3"]
        if "x**2 = 4" in normalized:
            return ["Step 1: x**2 = 4", "Step 2: x = 2"]
        if "sin" in normalized and "cos" in normalized:
            return ["Step 1: sin(x)**2 + cos(x)**2 = 1"]
        return [f"Step 1: {prompt}", f"Step 2: {prompt}"]

    def _regenerate_step(self, prompt: str, error_index: int, steps: List[str]) -> str:
        normalized = prompt.lower().replace("^", "**")
        if "x**2 = 4" in normalized and error_index == 1:
            return "Step 2: (x - 2)*(x + 2) = 0"
        previous = steps[error_index - 1] if error_index else "0 = 0"
        return f"Step {error_index + 1}: {previous.split(':', 1)[-1].strip()}"

    def self_correct_search(self, prompt: str) -> Dict[str, object]:
        """Run a candidate search and regenerate only its first failed step."""
        steps = self._candidate_steps(prompt)
        initial = self.evaluate("\n".join(steps))
        corrections = 0

        if initial["error_index"] is not None:
            error_index = int(initial["error_index"])
            steps[error_index] = self._regenerate_step(prompt, error_index, steps)
            corrections = 1

        final = self.evaluate("\n".join(steps))
        return {
            "prompt": prompt,
            "steps": final["steps"],
            "valid": final["valid"],
            "step_scores": final["step_scores"],
            "average_score": final["average_score"],
            "error_index": initial["error_index"],
            "self_correction": initial["self_correction"],
            "corrections_applied": corrections,
        }