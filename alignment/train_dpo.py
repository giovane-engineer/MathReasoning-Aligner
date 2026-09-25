"""
Direct Preference Optimization (DPO) pipeline to penalize mathematical hallucinations.
"""
from trl import DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import Dataset

def train_dpo():
    model_id = "Qwen/Qwen2.5-Math-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    ref_model = AutoModelForCausalLM.from_pretrained(model_id)

    # Dataset de preferências (chosen vs rejected)
    dpo_data = {
        "prompt": ["Compute d/dx(x^2)"],
        "chosen": ["Step 1: Use power rule d/dx(x^n) = n*x^(n-1). Step 2: 2*x^(2-1) = 2x."],
        "rejected": ["The derivative is x^3 / 3."]  # Resposta errada/alucinada
    }
    dataset = Dataset.from_dict(dpo_data)

    training_args = TrainingArguments(
        output_dir="./dpo_output",
        per_device_train_batch_size=1,
        learning_rate=5e-7,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    print("Starting DPO preference alignment...")
    # trainer.train() # Descomentar durante a execução real com GPU

if __name__ == "__main__":
    train_dpo()
