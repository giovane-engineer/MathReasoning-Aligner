import gradio as gr
from data.generator import MathDataGenerator
from data.validator import MathValidator

generator = MathDataGenerator()
validator = MathValidator()

def generate_and_validate(domain):
    # Gera o problema e o raciocínio CoT
    sample = generator.generate_sample(domain=domain)
    instruction = sample["instruction"]
    cot = sample["cot_reasoning"]
    ground_truth = sample["ground_truth"]
    
    # Valida a resposta
    is_valid, msg = validator.validate(ground_truth, ground_truth)
    
    return instruction, cot, ground_truth, f"Status da Validação: {'✅ VÁLIDO' if is_valid else '❌ INVÁLIDO'} ({msg})"

with gr.Blocks(title="MathReasoning-Aligner") as demo:
    gr.Markdown("# 🧮 MathReasoning-Aligner: Alignment & Symbolic Verification")
    gr.Markdown("Demonstração interativa de geração de dados sintéticos CoT e validação determinística via SymPy.")
    
    with gr.Row():
        domain_dropdown = gr.Dropdown(
            choices=["calculus", "differential_equations", "linear_algebra"],
            value="calculus",
            label="Domínio Matemático"
        )
        btn = gr.Button("Gerar e Validar", variant="primary")
    
    with gr.Column():
        out_instruction = gr.Textbox(label="Instrução / Problema")
        out_cot = gr.Textbox(label="Raciocínio Passo a Passo (CoT)", lines=5)
        out_gt = gr.Textbox(label="Ground Truth (SymPy)")
        out_val = gr.Textbox(label="Resultado do Validador Híbrido")
        
    btn.click(
        fn=generate_and_validate,
        inputs=[domain_dropdown],
        outputs=[out_instruction, out_cot, out_gt, out_val]
    )

demo.launch()