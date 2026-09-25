import json
from typing import Dict

import sympy as sp


SUPPORTED_DOMAINS = ("calculus", "differential_equations", "linear_algebra")


def _calculus_sample() -> Dict[str, str]:
    x = sp.Symbol('x')
    func = x**2 * sp.exp(x)
    result = sp.integrate(func, x)

    return {
        "domain": "calculus",
        "instruction": f"Compute the integral of $f(x) = {sp.latex(func)}$.",
        "cot_reasoning": (
            f"Step 1: Set u = x^2 and dv = e^x dx.\n"
            f"Step 2: Integration by parts gives x^2 e^x - 2 integral(x e^x dx).\n"
            f"Step 3: Apply integration by parts to the remaining integral.\n"
            f"Step 4: Therefore the antiderivative is ${sp.latex(result)} + C$."
        ),
        "ground_truth": str(result),
    }


def _differential_equations_sample() -> Dict[str, str]:
    x = sp.Symbol('x')
    solution = sp.exp(2 * x)

    return {
        "domain": "differential_equations",
        "instruction": "Solve y'(x) = 2*y(x) with initial condition y(0) = 1.",
        "cot_reasoning": (
            "Step 1: Separate variables: dy/y = 2 dx.\n"
            "Step 2: Integrate both sides: ln|y| = 2x + C.\n"
            "Step 3: Exponentiate: y = A*exp(2x).\n"
            "Step 4: Use y(0) = 1, so A = 1 and the solution is "
            f"$y(x) = {sp.latex(solution)}$."
        ),
        "ground_truth": str(solution),
    }


def _linear_algebra_sample() -> Dict[str, str]:
    matrix = sp.Matrix([[2, 1], [1, 3]])
    determinant = matrix.det()

    return {
        "domain": "linear_algebra",
        "instruction": f"Compute the determinant of A = {matrix.tolist()}.",
        "cot_reasoning": (
            "Step 1: For a 2x2 matrix, det(A) = ad - bc.\n"
            "Step 2: Substitute the entries: det(A) = 2*3 - 1*1.\n"
            f"Step 3: Therefore det(A) = {determinant}."
        ),
        "ground_truth": str(determinant),
    }


def generate_synthetic_cot_sample(domain: str = "calculus") -> Dict[str, str]:
    """Generate one deterministic chain-of-thought sample for a domain."""
    generators = {
        "calculus": _calculus_sample,
        "differential_equations": _differential_equations_sample,
        "linear_algebra": _linear_algebra_sample,
    }
    try:
        return generators[domain]()
    except KeyError as error:
        supported = ", ".join(SUPPORTED_DOMAINS)
        raise ValueError(f"Unsupported domain: {domain}. Choose one of: {supported}") from error


def generate_dpo_pair(domain: str = "calculus") -> Dict[str, str]:
    """Generate a DPO prompt with a rigorous answer and subtle hallucination."""
    sample = generate_synthetic_cot_sample(domain)
    rejected_answers = {
        "calculus": "By the power rule, the integral is exp(x)*x**2 + C.",
        "differential_equations": "Separating variables gives y(x) = exp(x), which satisfies the initial condition.",
        "linear_algebra": "The determinant is 2*3 + 1*1 = 7.",
    }
    return {
        "domain": sample["domain"],
        "prompt": sample["instruction"],
        "chosen": sample["cot_reasoning"] + f"\nAnswer: {sample['ground_truth']}",
        "rejected": rejected_answers[domain],
    }


if __name__ == "__main__":
    sample = generate_synthetic_cot_sample()
    print(json.dumps(sample, indent=2))