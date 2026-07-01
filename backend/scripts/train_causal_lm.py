"""
Pure document (causal LM) training on legal text. No Q&A data.
Raw text -> tokenize in blocks -> train to predict next token.
~4h on RTX 4090 for 7B; use --use-lora and/or smaller model if needed.
"""
import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from app.config import CAUSAL_LM_DIR as OUTPUT_DIR, CORPUS_PATH, get_env

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
BLOCK_SIZE = 1024


def load_corpus(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Corpus not found: {path}. Run: python build_raw_corpus.py")
    return path.read_text(encoding="utf-8")


def tokenize_raw(tokenizer, corpus: str, block_size: int) -> list:
    tokenizer.pad_token = tokenizer.eos_token
    raw = tokenizer(
        corpus,
        return_tensors=None,
        truncation=False,
        add_special_tokens=True,
    )
    input_ids = raw["input_ids"]
    blocks = []
    for i in range(0, len(input_ids) - block_size, block_size):
        blocks.append(input_ids[i : i + block_size])
    return blocks


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Causal LM training on raw tax corpus.")
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH, help="Path to tax_corpus.txt")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="Output model dir")
    parser.add_argument("--model", type=str, default=get_env("HF_MODEL", DEFAULT_MODEL), help="Base model")
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE, help="Token block size")
    parser.add_argument("--epochs", type=int, default=2, help="Epochs")
    parser.add_argument("--batch", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--use-lora", action="store_true", help="Use LoRA to reduce VRAM (for 7B on 16GB)")
    args = parser.parse_args()

    corpus = load_corpus(args.corpus)
    print(f"Corpus length: {len(corpus):,} chars")

    model_id = args.model
    print(f"Loading tokenizer and model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    if args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    blocks = tokenize_raw(tokenizer, corpus, args.block_size)
    n_blocks = len(blocks)
    print(f"Tokenized into {n_blocks} blocks of {args.block_size} tokens")

    from datasets import Dataset
    dataset = Dataset.from_dict({"input_ids": blocks})

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        learning_rate=args.lr,
        logging_steps=20,
        save_strategy="epoch",
        bf16=(device == "cuda"),
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
    print(f"Model saved to {args.output.resolve()}")
    print("Prompt template (post-training): User: <question>\\nAct: [model generates] -> Explain in plain English:")


if __name__ == "__main__":
    main()
