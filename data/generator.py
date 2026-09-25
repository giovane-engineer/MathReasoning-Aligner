import json
import sympy as sp


def generate_synthetic_cot_sample():
    x = sp.Symbol('x')
    func = sp.sin(x) * sp.exp(x)
    derivative = sp.diff(func, x)

    sample = {
        "instruction": f"Compute the derivative of $f(x) = {sp.latex(func)}$. Provide step-by-step reasoning.",
        "cot_reasoning": (
            f"Step 1: Identify $f(x) = {sp.latex(func)}$.\n"
            f"Step 2: Apply product rule: $(uv)' = u'v + uv'$.\n"
            f"Step 3: Derivatives: $\\frac{{d}}{{dx}}\\sin(x) = \\cos(x)$, $\\frac{{d}}{{dx}}e^x = e^x$.\n"
            f"Step 4: Combine to get $f'(x) = {sp.latex(derivative)}$."
        ),
        "ground_truth": str(derivative)
    }
    return sample


if __name__ == "__main__":
    sample = generate_synthetic_cot_sample()
    print(json.dumps(sample, indent=2))