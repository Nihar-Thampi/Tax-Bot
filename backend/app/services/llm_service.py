import os

# Avoid "duplicate template name" when loading PEFT adapter (bitsandbytes/torch.compile)
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from app.config import HF_MODEL, LOCAL_LLM_4BIT, LOCAL_LLM_DEVICE, LOCAL_LLM_MAX_CONTEXT, TAX_MODEL_ADAPTER
from app.services.cancellation import current_cancel_event, raise_if_cancelled

_model = None
_tokenizer = None


def _get_device():
    import torch
    device_override = LOCAL_LLM_DEVICE.lower()
    if device_override in ("cuda", "gpu", "cuda:0"):
        if device_override == "gpu":
            device_override = "cuda"
        return device_override
    if device_override in ("mps", "cpu"):
        return device_override
    if torch.cuda.is_available():
        return "cuda"
    # Apple Silicon GPU (macOS)
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_model_and_tokenizer():
    global _model, _tokenizer
    if _model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        device = _get_device()
        if device == "cpu" and not torch.cuda.is_available():
            print("CUDA not available (PyTorch may be CPU-only). For GPU: install PyTorch with CUDA, or set env LOCAL_LLM_DEVICE=cuda to force.")
        if device == "mps" and LOCAL_LLM_4BIT:
            print("4-bit loading needs an NVIDIA GPU; loading in float16 on Apple Silicon instead (~2 bytes per parameter).")
        # float16 halves memory vs float32: a 7B model needs ~15 GB instead of ~30 GB.
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        adapter_path = TAX_MODEL_ADAPTER
        if adapter_path and not os.path.isdir(adapter_path):
            adapter_path = ""
        quantize = LOCAL_LLM_4BIT and device.startswith("cuda")
        print(f"Loading Hugging Face model: {HF_MODEL}{' (4-bit)' if quantize else ''} ...")
        tokenizer = AutoTokenizer.from_pretrained(adapter_path or HF_MODEL, trust_remote_code=True)
        load_kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
        if quantize:
            # 4-bit weights: a 7B model fits in ~6 GB of VRAM.
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
            )
            load_kwargs["device_map"] = device
        else:
            load_kwargs["dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(HF_MODEL, **load_kwargs)
        if adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
            if not quantize:  # merging into 4-bit weights isn't supported
                model = model.merge_and_unload()
            print(f"Loaded fine-tuned adapter from {adapter_path}")
        if not quantize:
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


def _encode_chat(tokenizer, chat: list[dict[str, str]]) -> list[int]:
    """Token ids for a chat, ending with the assistant generation prompt."""
    if hasattr(tokenizer, "apply_chat_template"):
        ids = tokenizer.apply_chat_template(chat, add_generation_prompt=True)
        # Newer transformers versions return a BatchEncoding here instead of a bare list.
        if hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        return list(ids)
    prompt = "\n".join(
        f"{x['role'].capitalize()}: {x['content']}" for x in chat
    ) + "\nAssistant: "
    return tokenizer(prompt)["input_ids"]


def context_window() -> int:
    """Total tokens (prompt + reply) the loaded model can attend to."""
    model, _ = _get_model_and_tokenizer()
    native = getattr(model.config, "model_max_length", None) or getattr(
        model.config, "max_position_embeddings", 2048
    )
    # Long-context models (e.g. 32k) are capped to keep prompts fast and VRAM use low.
    return min(native, LOCAL_LLM_MAX_CONTEXT) if LOCAL_LLM_MAX_CONTEXT else native


def count_tokens(messages: list[dict[str, str]]) -> int:
    """Prompt length in tokens for these messages, as generate() would build it."""
    _, tokenizer = _get_model_and_tokenizer()
    return len(_encode_chat(tokenizer, _messages_to_chat(messages)))


def text_tokens(text: str) -> int:
    _, tokenizer = _get_model_and_tokenizer()
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def truncate_text_tokens(text: str, max_tokens: int) -> str:
    """Cut text down to at most max_tokens tokens."""
    _, tokenizer = _get_model_and_tokenizer()
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max(max_tokens, 0)], skip_special_tokens=True)


def generate(
    messages: list[dict[str, str]],
    max_tokens: int = 2048,
    temperature: float = 0.2,
    do_sample: bool = True,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> str:
    """
    Generate a reply from the local LLM given chat messages.
    If the prompt is too long, the start is dropped (not the end) so the latest
    question and the assistant prompt always reach the model.

    messages: list of {"role": "system"|"user"|"assistant", "content": "..."}
    """
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    model, tokenizer = _get_model_and_tokenizer()
    # The user may have pressed stop while the model was loading.
    raise_if_cancelled()
    chat = _messages_to_chat(messages)

    model_max_length = context_window()
    min_input_reserved = 512
    max_new_tokens = min(max_tokens, model_max_length - min_input_reserved)
    max_input_tokens = model_max_length - max_new_tokens

    ids = _encode_chat(tokenizer, chat)
    if len(ids) > max_input_tokens:
        print(
            f"Warning: prompt is {len(ids)} tokens but only {max_input_tokens} fit; "
            "dropping the start so the question is kept."
        )
        ids = ids[-max_input_tokens:]
    inputs = torch.tensor([ids], device=model.device, dtype=torch.long)

    input_len = inputs.shape[-1]
    attention_mask = torch.ones_like(inputs, device=model.device, dtype=torch.long)

    class _StopOnCancel(StoppingCriteria):
        # Checked after every generated token.
        def __call__(self, input_ids, scores, **kwargs):
            stop = cancel_event is not None and cancel_event.is_set()
            return torch.full((input_ids.shape[0],), stop, dtype=torch.bool, device=input_ids.device)

    cancel_event = current_cancel_event()

    with torch.no_grad():
        out = model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            do_sample=do_sample,
            top_p=0.95,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([_StopOnCancel()]),
        )
    raise_if_cancelled()

    reply = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
    return reply.strip()
