"""CPU-friendly Gradio interface for symbolic CoT search and PRM scoring."""

import json
from typing import Tuple

from alignment.prm_search import StepLevelPRM


EXAMPLE_PROBLEMS = [
    "Solve x + 2 = 5",
    "Solve x**2 = 4",
    "Simplify sin(x)**2 + cos(x)**2",
]


def generate_solution(problem: str) -> Tuple[str, dict, dict]:
    """Generate a symbolic solution and return validation plus PRM details."""
    if not problem or not problem.strip():
        empty = {"valid": False, "message": "Enter a mathematical problem."}
        return "", empty, empty

    result = StepLevelPRM().self_correct_search(problem.strip())
    solution = "\n".join(result["steps"])
    validation = {
        "valid": result["valid"],
        "corrections_applied": result["corrections_applied"],
        "error_index": result["error_index"],
    }
    prm = {
        "average_score": result["average_score"],
        "step_scores": result["step_scores"],
        "self_correction": result["self_correction"],
    }
    return solution, validation, prm


def build_app():
    """Build the Gradio app lazily so importing this module needs no Gradio."""
    import gradio as gr

    with gr.Blocks(title="MathReasoning Step-Level PRM") as demo:
        gr.Markdown("# MathReasoning Step-Level PRM\nSymbolic test-time search with self-correction.")
        problem = gr.Textbox(
            label="Mathematical problem",
            placeholder="Try: Solve x**2 = 4",
        )
        gr.Examples(EXAMPLE_PROBLEMS, inputs=problem)
        generate = gr.Button("Generate and validate", variant="primary")
        solution = gr.Textbox(label="CoT solution", lines=8)
        validation = gr.JSON(label="Symbolic validation")
        prm = gr.JSON(label="Step-level PRM scores")

        generate.click(generate_solution, inputs=problem, outputs=[solution, validation, prm])
        problem.change(generate_solution, inputs=problem, outputs=[solution, validation, prm])

    return demo


if __name__ == "__main__":
    build_app().launch()