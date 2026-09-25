import sympy as sp

def verify_symbolic_equivalence(prediction: str, target: str) -> bool:
    """
    Verifica se a resposta do LLM é matematicamente equivalente ao target
    utilizando simplificação simbólica do SymPy.
    """
    try:
        pred_expr = sp.sympify(prediction)
        target_expr = sp.sympify(target)
        return sp.simplify(pred_expr - target_expr) == 0
    except Exception:
        return False

if __name__ == "__main__":
    pred = "exp(x)*sin(x) + exp(x)*cos(x)"
    target = "exp(x)*(sin(x) + cos(x))"
    print(f"Mathematical Match: {verify_symbolic_equivalence(pred, target)}")
