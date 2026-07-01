import os

# Avoid "duplicate template name" when loading PEFT adapter (bitsandbytes/torch.compile)
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from typing import Any

from env_config import get_env

_DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
_model = None
_tokenizer = None


def _get_device():
    import torch
    device_override = get_env("LOCAL_LLM_DEVICE").lower()
    if device_override in ("cuda", "gpu", "cuda:0"):
        if device_override == "gpu":
            device_override = "cuda"
        return device_override
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_model_and_tokenizer():
    global _model, _tokenizer
    if _model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        device = _get_device()
        if device == "cpu" and not torch.cuda.is_available():
            print("CUDA not available (PyTorch may be CPU-only). For GPU: install PyTorch with CUDA, or set env LOCAL_LLM_DEVICE=cuda to force.")
        dtype = torch.float16 if device == "cuda" else torch.float32
        model_id = get_env("HF_MODEL", _DEFAULT_MODEL)
        adapter_path = get_env("TAX_MODEL_ADAPTER")
        if adapter_path and not os.path.isdir(adapter_path):
            adapter_path = ""
        print(f"Loading Hugging Face model: {model_id} ...")
        tokenizer = AutoTokenizer.from_pretrained(adapter_path or model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        if adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()
            print(f"Loaded fine-tuned adapter from {adapter_path}")
        model = model.to(device)
        model.eval()
        print(f"Model loaded on {device}.")
        _model, _tokenizer = model, tokenizer
    return _model, _tokenizer


def _messages_to_chat(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Normalise to role/content only; merge system into first user if needed."""
    out = []
    system_parts = []
    for m in messages:
        role = (m.get("role") or "user").strip().lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            system_parts.append(content)
            continue
        if role == "user" and system_parts:
            content = "\n\n".join(system_parts) + "\n\n" + content
            system_parts = []
        out.append({"role": role, "content": content})
    if system_parts and out:
        out[0]["content"] = "\n\n".join(system_parts) + "\n\n" + out[0]["content"]
    return out if out else [{"role": "user", "content": "Hello."}]


def generate(
    messages: list[dict[str, str]],
    max_tokens: int = 2048,
    temperature: float = 0.2,
    do_sample: bool = True,
) -> str:
    """
    Generate a reply from the local LLM given chat messages.
    Input is truncated to the model's max length so generation never hangs.

    messages: list of {"role": "system"|"user"|"assistant", "content": "..."}
    """
    import torch

    model, tokenizer = _get_model_and_tokenizer()
    chat = _messages_to_chat(messages)

    model_max_length = getattr(
        model.config, "model_max_length", None
    ) or getattr(model.config, "max_position_embeddings", 2048)
    min_input_reserved = 512
    max_new_tokens = min(max_tokens, model_max_length - min_input_reserved)
    max_input_tokens = model_max_length - max_new_tokens

    if hasattr(tokenizer, "apply_chat_template"):
        inputs = tokenizer.apply_chat_template(
            chat,
            return_tensors="pt",
            add_generation_prompt=True,
            truncation=True,
            max_length=max_input_tokens,
        )
        inputs = inputs.to(model.device)
    else:
        prompt = "\n".join(
            f"{x['role'].capitalize()}: {x['content']}" for x in chat
        ) + "\nAssistant: "
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        )
        inputs = enc["input_ids"].to(model.device)

    input_len = inputs.shape[-1]
    attention_mask = torch.ones_like(inputs, device=model.device, dtype=torch.long)

    with torch.no_grad():
        out = model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            do_sample=do_sample,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    reply = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
    return reply.strip()
