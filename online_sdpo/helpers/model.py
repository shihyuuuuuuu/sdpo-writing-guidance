import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)


def load_model_and_tokenizer(model_name: str, gpu: str, attn: str = "sdpa"):
    # 4-bit NF4 weights with bf16 computation
    user_bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    device_map = {"": f"cuda:{gpu}"}
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=user_bnb_config,
        dtype=torch.bfloat16,
        attn_implementation=attn,
        device_map=device_map,
    )
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side="left", truncation_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    return model, tok

@torch.no_grad()
def generate_chat_batch(
    model,
    tokenizer,
    device,
    chats,
    max_input_tokens,
    max_new_tokens,
    temperature,
    top_p=0.9,
):
    prompt_texts = [
        tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for chat in chats
    ]

    enc = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
        add_special_tokens=False,
    ).to(device)

    prompt_len = enc["input_ids"].shape[-1]
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature > 0:
        gen_kwargs.update(
            temperature=temperature,
            top_p=top_p,
        )
    gen_config = GenerationConfig(**gen_kwargs)

    output_ids = model.generate(**enc, generation_config=gen_config)
    generated_ids = output_ids[:, prompt_len:]
    return [
        text.strip()
        for text in tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
    ]
