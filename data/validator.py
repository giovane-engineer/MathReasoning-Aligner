"""Hybrid exact and numerical validation for symbolic model outputs."""

import random
import signal
from contextlib import contextmanager
from typing import Iterator, Sequence

import sympy as sp


class SymbolicValidationTimeout(TimeoutError):
    """Raised when exact symbolic validation exceeds its time budget."""


@contextmanager
def _time_limit(seconds: float) -> Iterator[None]:
    def handle_timeout(signum, frame):
        raise SymbolicValidationTimeout("Symbolic validation timed out")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _parse_expression(expression: str):
    return sp.sympify(expression, locals={"Matrix": sp.Matrix})


def _exactly_equivalent(prediction, target) -> bool:
    if isinstance(prediction, sp.MatrixBase) or isinstance(target, sp.MatrixBase):
        if not isinstance(prediction, sp.MatrixBase) or not isinstance(target, sp.MatrixBase):
            return False
        if prediction.shape != target.shape:
            return False
        return all(sp.simplify(value) == 0 for value in prediction - target)
    return sp.simplify(prediction - target) == 0


def _finite_number(value) -> bool:
    try:
        return bool(sp.N(value).is_finite)
    except (TypeError, ValueError):
        return False


def _numeric_values(expression, points: Sequence[dict]):
    values = []
    for point in points:
        evaluated = expression.subs(point)
        entries = list(evaluated) if isinstance(evaluated, sp.MatrixBase) else [evaluated]
        if not all(_finite_number(value) for value in entries):
            raise ValueError("Expression is undefined at a sampled point")
        values.append([complex(sp.N(value)) for value in entries])
    return values


def _numerically_equivalent(prediction, target, samples: int = 15) -> bool:
    symbols = sorted(prediction.free_symbols | target.free_symbols, key=lambda item: item.name)
    rng = random.Random(20260925)
    points = [
        {symbol: sp.Rational(rng.randint(-30, 30), 10) for symbol in symbols}
        for _ in range(samples)
    ]
    if not symbols:
        points = [{}]

    try:
        prediction_values = _numeric_values(prediction, points)
        target_values = _numeric_values(target, points)
    except (TypeError, ValueError, ZeroDivisionError):
        return False

    return all(
        len(predicted) == len(expected)
        and all(
            abs(left - right) <= 1e-9 * max(1.0, abs(left), abs(right))
            for left, right in zip(predicted, expected)
        )
        for predicted, expected in zip(prediction_values, target_values)
    )


def verify_symbolic_equivalence(
    prediction: str,
    target: str,
    *,
    exact_timeout: float = 2.0,
    monte_carlo_samples: int = 15,
) -> bool:
    """Verify equivalence exactly first, then with deterministic numeric samples."""
    try:
        predicted_expr = _parse_expression(prediction)
        target_expr = _parse_expression(target)
    except (TypeError, SyntaxError, ValueError, sp.SympifyError):
        return False

    try:
        with _time_limit(exact_timeout):
            if _exactly_equivalent(predicted_expr, target_expr):
                return True
    except (SymbolicValidationTimeout, TypeError, ValueError, ArithmeticError):
        pass

    return _numerically_equivalent(predicted_expr, target_expr, monte_carlo_samples)


if __name__ == "__main__":
    prediction = "exp(x)*sin(x) + exp(x)*cos(x)"
    target = "exp(x)*(sin(x) + cos(x))"
    print(f"Mathematical Match: {verify_symbolic_equivalence(prediction, target)}")