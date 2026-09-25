"""
Deterministic Evaluation Benchmark comparing predictions with SymPy ground-truth.
"""
from data.validator import verify_symbolic_equivalence

def run_math_benchmark():
    test_suite = [
        {"input": "d/dx(x^2)", "pred": "2*x", "target": "2*x"},
        {"input": "d/dx(sin(x))", "pred": "cos(x)", "target": "cos(x)"},
        {"input": "d/dx(exp(x))", "pred": "exp(x) + 1", "target": "exp(x)"}, # Erro propositado
    ]

    correct = 0
    total = len(test_suite)

    for test in test_suite:
        is_correct = verify_symbolic_equivalence(test["pred"], test["target"])
        if is_correct:
            correct += 1
        print(f"Input: {test['input']} | Correct: {is_correct}")

    accuracy = (correct / total) * 100
    print(f"\nBenchmark Precision: {accuracy:.2f}%")

if __name__ == "__main__":
    run_math_benchmark()
