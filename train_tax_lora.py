"""
Fine-tune the local LLM (e.g. TinyLlama) on the SA Income Tax Act with LoRA.
Run: python build_tax_training_data.py  then  python train_tax_lora.py
Then set TAX_MODEL_ADAPTER to the output folder (e.g. tax_lora_adapter) so the chatbot uses the fine-tuned model.
"""
import os

# Avoid bitsandbytes/torch.compile triggering "duplicate template name" in PyTorch inductor
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

from env_config import get_env

DATASET_PATH = Path(__file__).resolve().parent / "tax_training_data.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "tax_lora_adapter"
DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_LENGTH = 768


def load_examples(path: Path) -> list[dict]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def format_chat(instruction: str, input_text: str, output: str) -> list[dict]:
    user_content = f"{instruction}\n\n{input_text}".strip()
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output},
    ]


def tokenize_and_mask(examples, tokenizer):
    """Tokenize instruction examples and create labels (-100 on user part, real ids on assistant)."""
    all_input_ids = []
    all_labels = []
    for i in range(len(examples["instruction"])):
        inst = examples["instruction"][i]
        inp = examples["input"][i]
        out = examples["output"][i]
        messages = format_chat(inst, inp, out)
        if hasattr(tokenizer, "apply_chat_template"):
            full = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            tokenized = tokenizer(
                full,
                truncation=True,
                max_length=MAX_LENGTH,
                padding=False,
                return_tensors=None,
            )
            input_ids = tokenized["input_ids"]
            # Mask user turn: find where assistant starts (after the last user turn token pattern)
            # For many chat templates, we can build user-only and full; labels = -100 for user part
            user_only = tokenizer.apply_chat_template(
                [messages[0]],
                tokenize=False,
                add_generation_prompt=True,
            )
            user_ids = tokenizer(user_only, return_tensors=None)["input_ids"]
            len_user = len(user_ids)
            labels = [-100] * min(len_user, len(input_ids)) + list(input_ids[min(len_user, len(input_ids)):])
            if len(labels) < len(input_ids):
                labels += [-100] * (len(input_ids) - len(labels))
            else:
                labels = labels[: len(input_ids)]
        else:
            text = f"User: {messages[0]['content']}\nAssistant: {messages[1]['content']}"
            tokenized = tokenizer(
                text,
                truncation=True,
                max_length=MAX_LENGTH,
                padding=False,
                return_tensors=None,
            )
            input_ids = tokenized["input_ids"]
            user_text = f"User: {messages[0]['content']}\nAssistant: "
            user_ids = tokenizer(user_text, return_tensors=None)["input_ids"]
            len_user = len(user_ids)
            labels = [-100] * min(len_user, len(input_ids)) + list(input_ids[len_user:])
            if len(labels) < len(input_ids):
                labels += [-100] * (len(input_ids) - len(labels))
            else:
                labels = labels[: len(input_ids)]
        all_input_ids.append(input_ids)
        all_labels.append(labels)
    return {"input_ids": all_input_ids, "labels": all_labels}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="LoRA fine-tune the LLM on SA tax Act data.")
    parser.add_argument("--data", type=Path, default=DATASET_PATH, help="Path to tax_training_data.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="Where to save the LoRA adapter")
    parser.add_argument("--model", type=str, default=get_env("HF_MODEL", DEFAULT_MODEL), help="Base model name")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=1, help="Per-device batch size (use 1 to avoid OOM)")
    parser.add_argument("--max-examples", type=int, default=80, help="Use at most this many examples (keeps training short and light)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    args = parser.parse_args()

    if not args.data.exists():
        print(f"Dataset not found: {args.data}. Run: python build_tax_training_data.py")
        return

    model_id = args.model
    print(f"Loading tokenizer and model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device_override = get_env("TAX_TRAIN_DEVICE").lower()
    if device_override in ("cuda", "gpu"):
        device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: training on CPU (slow). Set TAX_TRAIN_DEVICE=cuda to force GPU if available.")
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)

    lora_config = LoraConfig(
        r=4,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    examples = load_examples(args.data)
    if args.max_examples and args.max_examples > 0 and len(examples) > args.max_examples:
        import random
        random.seed(42)
        examples = random.sample(examples, args.max_examples)
    print(f"Training on {len(examples)} examples")
    if not examples:
        print("No examples. Run build_tax_training_data.py first.")
        return

    # Build lists for tokenize
    data = {
        "instruction": [e["instruction"] for e in examples],
        "input": [e["input"] for e in examples],
        "output": [e["output"] for e in examples],
    }
    tokenized = tokenize_and_mask(data, tokenizer)
    from datasets import Dataset
    dataset = Dataset.from_dict({
        "input_ids": tokenized["input_ids"],
        "labels": tokenized["labels"],
    })

    def _collate(batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        input_ids = []
        labels = []
        for b in batch:
            ids = b["input_ids"]
            labs = b["labels"]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_len)
            labels.append(labs + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    collator = _collate

    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        logging_steps=5,
        save_strategy="epoch",
        bf16=(device == "cuda"),
        fp16=False,
        remove_unused_columns=False,
        max_grad_norm=1.0,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print(f"Adapter and tokenizer saved to {args.output.resolve()}")
    print("To use: set env TAX_MODEL_ADAPTER to this path, then run the chatbot or scanner.")


if __name__ == "__main__":
    main()
