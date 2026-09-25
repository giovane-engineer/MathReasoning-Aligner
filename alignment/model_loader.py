"""Shared model-backbone loader with optional 4-bit QLoRA support."""

from typing import List, Tuple


SUPPORTED_BACK_MODELS = (
    "Qwen/Qwen2.5-Math-1.5B",
    "deepseek-ai/deepseek-math-7b-instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
)

LORA_RANK = 16
LORA_ALPHA = 32


def _find_lora_targets(model) -> List[str]:
    candidates = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    targets = {
        name.rsplit(".", 1)[-1]
        for name, _ in model.named_modules()
        if name.rsplit(".", 1)[-1] in candidates
    }
    if not targets:
        raise ValueError("Could not find compatible attention or MLP projection layers for LoRA")
    return sorted(targets)


def load_back_model(model_name: str, quantize_4bit: bool = True) -> Tuple[object, object]:
    """Load a supported causal LM and tokenizer prepared for LoRA fine-tuning.

    The model is loaded with NF4 quantization by default. Set ``quantize_4bit``
    to ``False`` for full-precision loading on hardware without BitsAndBytes.
    """
    if model_name not in SUPPORTED_BACK_MODELS:
        supported = ", ".join(SUPPORTED_BACK_MODELS)
        raise ValueError(f"Unsupported back model: {model_name}. Choose one of: {supported}")

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_kwargs = {"device_map": "auto"}
    if quantize_4bit:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as error:
            raise ImportError(
                "4-bit loading requires bitsandbytes. Install it with "
                "'pip install bitsandbytes', or pass quantize_4bit=False."
            ) from error

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    if quantize_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=_find_lora_targets(model),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer