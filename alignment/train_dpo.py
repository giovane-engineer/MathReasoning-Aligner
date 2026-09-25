"""
Direct Preference Optimization (DPO) pipeline to penalize mathematical hallucinations.
"""
import argparse

from trl import DPOTrainer
from transformers import TrainingArguments
from datasets import Dataset

try:
    from alignment.model_loader import SUPPORTED_BACK_MODELS, load_back_model
except ModuleNotFoundError:
    from model_loader import SUPPORTED_BACK_MODELS, load_back_model


def train_dpo(model_name: str = SUPPORTED_BACK_MODELS[0], quantize_4bit: bool = True):
    model, tokenizer = load_back_model(model_name, quantize_4bit=quantize_4bit)

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
        gradient_checkpointing=quantize_4bit,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    print("Starting DPO preference alignment...")
    # trainer.train() # Descomentar durante a execução real com GPU

def parse_args():
    parser = argparse.ArgumentParser(description="Run DPO with a selectable math back model.")
    parser.add_argument("--model", choices=SUPPORTED_BACK_MODELS, default=SUPPORTED_BACK_MODELS[0])
    parser.add_argument("--no-quantize-4bit", action="store_false", dest="quantize_4bit")
    parser.set_defaults(quantize_4bit=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_dpo(model_name=args.model, quantize_4bit=args.quantize_4bit)
