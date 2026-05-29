"""
Generate completions for extractor audit, independent of any training run.

Loads the base model, samples N problems from GSM8K, generates completions with
varied temperatures to maximize diversity (different formats, different failure
modes, different lengths). Writes a JSON file in the shape the audit script expects.

Usage:
    python generate_audit_sample.py --n 500 --output audit_sample.json
"""

import argparse
import json
import random

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = (
    "Solve the math problem step by step. Wrap your final answer in <answer> tags, e.g. <answer>42</answer>."
)


def build_prompt(tokenizer, question: str) -> str:
    """Apply the chat template the model was trained with."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_completions(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    n: int = 500,
    output_path: str = "audit_sample.json",
    seed: int = 42,
):
    random.seed(seed)
    torch.manual_seed(seed)

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    print("Loading GSM8K test split...")
    ds = load_dataset("openai/gsm8k", "main")["test"]
    # Shuffle so we get a random slice, not the first N
    indices = list(range(len(ds)))
    random.shuffle(indices)
    indices = indices[:n]

    # Mix of temperatures for diversity: low temp gives clean answers,
    # high temp gives messier completions (truncated, off-format, rambling)
    # — exactly the diversity we want the judge tested on.
    temperatures = [0.3, 0.7, 1.0, 1.3]

    results = []
    for i, idx in enumerate(indices):
        if i % 25 == 0:
            print(f"[{i}/{n}] generating...")

        example = ds[idx]
        question = example["question"]
        gt = example["answer"].split("####")[-1].strip()
        temp = temperatures[i % len(temperatures)]

        prompt_text = build_prompt(tokenizer, question)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=temp,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Strip the prompt, decode only the new tokens
        completion = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        results.append(
            {
                "problem": question,
                "gt": gt,
                "completion": completion,
                "temperature": temp,  # kept for diagnostics, not used by audit
            }
        )

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote {len(results)} completions to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--output", default="audit_sample.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_completions(
        model_name=args.model,
        n=args.n,
        output_path=args.output,
        seed=args.seed,
    )
