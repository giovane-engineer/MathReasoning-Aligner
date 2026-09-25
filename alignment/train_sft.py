"""
Supervised Fine-Tuning (SFT) script for Chain-of-Thought alignment.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import Dataset

def train_sft():
    model_id = "Qwen/Qwen2.5-Math-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

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
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        args=training_args,
    )

    print("Starting SFT training pipeline...")
    # trainer.train()  # Descomentar durante a execução real com GPU

if __name__ == "__main__":
    train_sft()
