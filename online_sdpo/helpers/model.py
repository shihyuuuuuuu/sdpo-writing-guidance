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
    top_p=0.8,
):
    prompt_texts = [
        tokenizer.apply_chat_template(  # 加上 <|im_start|>...<|im_end|>
            chat,
            tokenize=False,
            add_generation_prompt=True, # 最後加上 <|im_start|>assistant\n<think>\n
            enable_thinking=False,      # 加上 \n</think>\n\n，把 think 關起來
        )
        for chat in chats
    ]

    enc = tokenizer(                    # encodes a batch of text into `input_ids`, `attention_mask`
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
        add_special_tokens=False,       # 已經使用 `apply_chat_template`，避免重複加上標籤
    ).to(device)

    # _print_tokens_length(enc["input_ids"], tokenizer, "Input")
    prompt_len = enc["input_ids"].shape[-1]
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "presence_penalty": 0.2,
        "repetition_penalty": 1.1,
    }
    if temperature > 0:
        gen_kwargs.update(
            temperature=temperature,
            top_p=top_p,
            top_k=20,
            min_p=0,
        )
    gen_config = GenerationConfig(**gen_kwargs)

    output_ids = model.generate(**enc, generation_config=gen_config)
    generated_ids = output_ids[:, prompt_len:]
    # _print_tokens_length(generated_ids, tokenizer, "Output")
    return [
        text.strip()
        for text in tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
    ]

def _print_tokens_length(token_ids, tokenizer, io: str) -> None:
    actual_lengths = (token_ids != tokenizer.pad_token_id).sum(dim=-1).float()
    avg_len = actual_lengths.mean().item()
    max_len = int(actual_lengths.max().item())
    print(f"{io} tokens(batch_size={len(token_ids)}): Avg.:{avg_len}, Max:{max_len}")