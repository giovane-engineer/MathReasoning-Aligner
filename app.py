import gradio as gr
from data.generator import MathDataGenerator
from data.validator import MathValidator

generator = MathDataGenerator()
validator = MathValidator()

def generate_and_validate(domain):
    sample = generator.generate_sample(domain=domain)
    instruction = sample["instruction"]
    cot = sample["cot_reasoning"]
    ground_truth = sample["ground_truth"]
    
    is_valid, msg = validator.validate(ground_truth, ground_truth)
    return instruction, cot, ground_truth, f"Status: {'✅ VÁLIDO' if is_valid else '❌ INVÁLIDO'} ({msg})"

with gr.Blocks(title="MathReasoning-Aligner") as demo:
    gr.Markdown("# 🧮 MathReasoning-Aligner")
    gr.Markdown("Synthetic CoT Generation & Symbolic Verification")
    
    with gr.Row():
        domain_dropdown = gr.Dropdown(
            choices=["calculus", "differential_equations", "linear_algebra"],
            value="calculus",
            label="Domínio Matemático"
        )
        btn = gr.Button("Gerar e Validar", variant="primary")
    
    with gr.Column():
        out_instruction = gr.Textbox(label="Instrução")
        out_cot = gr.Textbox(label="Raciocínio CoT", lines=5)
        out_gt = gr.Textbox(label="Ground Truth")
        out_val = gr.Textbox(label="Validação SymPy")
        
    btn.click(
        fn=generate_and_validate,
        inputs=[domain_dropdown],
        outputs=[out_instruction, out_cot, out_gt, out_val]
    )

if __name__ == "__main__":
    demo.queue().launch()