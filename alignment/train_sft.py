"""
Supervised Fine-Tuning (SFT) script for Chain-of-Thought alignment.
"""
import argparse

from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset

try:
    from alignment.model_loader import SUPPORTED_BACK_MODELS, load_back_model
except ModuleNotFoundError:
    from model_loader import SUPPORTED_BACK_MODELS, load_back_model


def train_sft(model_name: str = SUPPORTED_BACK_MODELS[0], quantize_4bit: bool = True):
    model, tokenizer = load_back_model(model_name, quantize_4bit=quantize_4bit)

    # Exemplo de dataset para treino CoT
    data = {
        "text": [
            "Instruction: Solve d/dx(sin(x)*exp(x))\nReasoning: Step 1: Apply product rule...\nAnswer: exp(x)*(sin(x) + cos(x))"
        ]
    }
    dataset = Dataset.from_dict(data)

    training_args = TrainingArguments(
        output_dir="./sft_output",
        per_device_train_batch_size=2,
        num_train_epochs=1,
        logging_steps=10,
        learning_rate=2e-5,
        gradient_checkpointing=quantize_4bit,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        args=training_args,
        tokenizer=tokenizer,
    )

    print("Starting SFT training pipeline...")
    # trainer.train()  # Descomentar durante a execução real com GPU

def parse_args():
    parser = argparse.ArgumentParser(description="Run SFT with a selectable math back model.")
    parser.add_argument("--model", choices=SUPPORTED_BACK_MODELS, default=SUPPORTED_BACK_MODELS[0])
    parser.add_argument("--no-quantize-4bit", action="store_false", dest="quantize_4bit")
    parser.set_defaults(quantize_4bit=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_sft(model_name=args.model, quantize_4bit=args.quantize_4bit)
