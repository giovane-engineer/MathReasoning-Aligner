"""Gradio interface for synthetic CoT, symbolic validation and PRM search."""

import os
from typing import Tuple

import gradio as gr

from alignment.prm_search import StepLevelPRM
from data.generator import SUPPORTED_DOMAINS, generate_dpo_pair, generate_synthetic_cot_sample
from data.validator import verify_symbolic_equivalence


DOMAIN_LABELS = {
    "calculus": "Calculus",
    "differential_equations": "Differential equations",
    "linear_algebra": "Symbolic linear algebra",
}

PRM_PROMPTS = {
    "calculus": "Solve x**2 = 4",
    "differential_equations": "Solve x + 2 = 5",
    "linear_algebra": "Simplify sin(x)**2 + cos(x)**2",
}


def _domain_name(domain: str) -> str:
    return DOMAIN_LABELS.get(domain, domain)


def generate_cot(domain: str) -> str:
    sample = generate_synthetic_cot_sample(domain)
    return (
        f"### Synthetic CoT: {_domain_name(domain)}\n\n"
        f"**Problem:** {sample['instruction']}\n\n"
        f"{sample['cot_reasoning']}\n\n"
        f"**Ground truth:** $${sample['ground_truth']}$$"
    )


def validate_domain(domain: str) -> str:
    sample = generate_synthetic_cot_sample(domain)
    pair = generate_dpo_pair(domain)
    chosen_valid = verify_symbolic_equivalence(sample["ground_truth"], sample["ground_truth"])
    rejected_valid = verify_symbolic_equivalence(pair["rejected"], sample["ground_truth"])
    return (
        f"### SymPy Validation: {_domain_name(domain)}\n\n"
        f"- Chosen answer: {'PASS' if chosen_valid else 'FAIL'}\n"
        f"- Rejected answer: {'UNEXPECTED MATCH' if rejected_valid else 'REJECTED'}\n\n"
        f"**Canonical result:** $${sample['ground_truth']}$$"
    )


def run_prm_search(domain: str) -> str:
    prompt = PRM_PROMPTS[domain]
    result = StepLevelPRM().self_correct_search(prompt)
    steps = "\n".join(f"- {step}" for step in result["steps"])
    correction = result["self_correction"] or "No correction required."
    return (
        f"### Step-Level PRM Search: {_domain_name(domain)}\n\n"
        f"**Test prompt:** `{prompt}`\n\n"
        f"**Final steps**\n{steps}\n\n"
        f"**Average score:** `{result['average_score']:.2f}`  \n"
        f"**Corrections applied:** `{result['corrections_applied']}`  \n"
        f"**First error index:** `{result['error_index']}`  \n"
        f"**Self-correction:** {correction}"
    )


def generate_solution(problem: str) -> Tuple[str, dict, dict]:
    """Backward-compatible helper for the previous single-button interface."""
    result = StepLevelPRM().self_correct_search(problem.strip())
    return (
        "\n".join(result["steps"]),
        {"valid": result["valid"], "error_index": result["error_index"]},
        {"average_score": result["average_score"], "step_scores": result["step_scores"]},
    )


def build_app():
    """Build the Spaces-ready Gradio app without loading a language model."""
    with gr.Blocks(title="MathReasoning Aligner") as demo:
        gr.Markdown(
            "# MathReasoning Aligner\n"
            "Synthetic CoT generation, symbolic validation and Step-Level PRM Search."
        )
        with gr.Row():
            domain = gr.Dropdown(
                choices=list(SUPPORTED_DOMAINS),
                value="calculus",
                label="Domain",
            )
            problem = gr.Textbox(
                label="Optional problem for compatibility mode",
                placeholder="The domain buttons use deterministic SymPy examples.",
            )

        with gr.Row():
            cot_button = gr.Button("Generate Synthetic CoT", variant="primary")
            validation_button = gr.Button("Run SymPy Validation")
            prm_button = gr.Button("Run Step-Level PRM Search")

        cot_output = gr.Markdown(label="CoT generation")
        validation_output = gr.Markdown(label="Symbolic validation")
        prm_output = gr.Markdown(label="PRM search")

        cot_button.click(generate_cot, inputs=domain, outputs=cot_output)
        validation_button.click(validate_domain, inputs=domain, outputs=validation_output)
        prm_button.click(run_prm_search, inputs=domain, outputs=prm_output)

    return demo


demo = build_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)