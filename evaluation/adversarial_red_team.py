"""Adversarial stress tests for subtle symbolic reasoning failures."""

import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.validator import verify_symbolic_equivalence


ADVERSARIAL_CASES: List[Dict[str, str]] = [
    {
        "name": "division by a potentially zero term",
        "prediction": "x/x",
        "target": "1",
    },
    {
        "name": "logarithm identity outside the real domain",
        "prediction": "log(x**2)",
        "target": "2*log(x)",
    },
    {
        "name": "omitted secondary polynomial root",
        "prediction": "x - 2",
        "target": "x**2 - 4",
    },
    {
        "name": "cancellation that excludes x = 1",
        "prediction": "(x**2 - 1)/(x - 1)",
        "target": "x + 1",
    },
]


def run_adversarial_red_team() -> float:
    detected = 0
    for case in ADVERSARIAL_CASES:
        accepted = verify_symbolic_equivalence(case["prediction"], case["target"])
        caught = not accepted
        detected += int(caught)
        print(f"{case['name']}: {'DETECTED' if caught else 'MISSED'}")

    precision = detected / len(ADVERSARIAL_CASES) * 100
    print(f"\nAdversarial Detection: {precision:.2f}%")
    if precision != 100.0:
        raise AssertionError("The hybrid validator missed an adversarial case")
    return precision


if __name__ == "__main__":
    run_adversarial_red_team()