from __future__ import annotations

from typing import Iterable, Mapping

from src.exp2_1_prompt.metrics import remove_thought_chain
from src.exp2_1_prompt.prompts import build_prompt_variants


def smoke_generation(item: Mapping[str, object], variant_name: str) -> str:
    keywords = str(item["aspect"]).replace(" ", "、")
    sentiment = "积极" if item["sentiment"] == "正面" else "保留"
    sentence = (
        f"《{item['title']}》的整体观感保持{sentiment}态度，"
        f"围绕{keywords}展开评价，文本来自{variant_name}配置。"
    )
    return sentence * 8


def load_model(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="npu",
        trust_remote_code=True,
    )
    return model, tokenizer


def generate_once(model, tokenizer, prompt: str, max_input_length: int, max_new_tokens: int) -> str:
    import torch

    messages = [{"role": "system", "content": ""}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", max_length=max_input_length, truncation=True).to("npu")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return remove_thought_chain(response)


def run_prompt_variants(
    items: Iterable[Mapping[str, object]],
    *,
    model_path: str,
    run_mode: str,
    max_input_length: int = 8192,
    max_new_tokens: int = 2048,
) -> list[dict[str, object]]:
    model = tokenizer = None
    if run_mode != "smoke":
        model, tokenizer = load_model(model_path)

    rows: list[dict[str, object]] = []
    for item in items:
        for variant in build_prompt_variants(item):
            if run_mode == "smoke":
                generation = smoke_generation(item, variant.name)
            else:
                generation = generate_once(model, tokenizer, variant.prompt, max_input_length, max_new_tokens)
            rows.append(
                {
                    "title": item["title"],
                    "aspect": item["aspect"],
                    "sentiment": item["sentiment"],
                    "intensity": item["intensity"],
                    "prompt_variant": variant.name,
                    "factor": variant.factor,
                    "prompt": variant.prompt,
                    "generation": generation,
                }
            )
    return rows
