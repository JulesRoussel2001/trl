# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl",
# ]
# ///

"""Shared evaluation utilities for GRPO experiments 1, 2, 3."""

import re


# === Answer normalization ===
def normalize_answer(s: str) -> str:
    """Normalize a numeric answer string for comparison."""
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    s = s.replace(" ", "")  # "1 000" → "1000"
    s = s.lstrip("+")  # "+42" → "42"
    try:
        num = float(s)
        s = str(int(num)) if num == int(num) else str(num)
    except ValueError:
        pass
    return s


# === Lenient substring matching (the "reward" matcher, matches Exp 1) ===
def answer_in_text(gt: str, text: str) -> bool:
    """Lenient substring match — does GT number appear anywhere in text?"""
    gt = normalize_answer(gt)
    if not gt or not text:
        return False
    text_normalized = text.replace(",", "").replace("$", "").replace("%", "")
    pattern = r"(?<![\d.-])" + re.escape(gt) + r"(?!\d|\.[\d])"
    return re.search(pattern, text_normalized) is not None


# === Strict accuracy extraction (the honest metric) ===
def extract_answer_strict(text: str) -> str | None:
    """Extract model's intended final answer. Tiered: <answer> → \\boxed → last number."""

    def last_num(s):
        nums = re.findall(r"[\d,]+(?:\.\d+)?", s)
        return nums[-1].replace(",", "") if nums else None

    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m and last_num(m.group(1)) is not None:
        return last_num(m.group(1))
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m and last_num(m.group(1)) is not None:
        return last_num(m.group(1))
    return last_num(text)


def strict_correct(gt: str, text: str) -> bool:
    """Strict correctness: does extracted answer numerically equal GT?"""
    pred = extract_answer_strict(text)
    if pred is None:
        return False
    try:
        return float(pred) == float(normalize_answer(gt))
    except ValueError:
        return False


# === Format compliance (observational) ===
FORMAT_PATTERN = re.compile(r"<answer>.*?</answer>", re.DOTALL)


def has_answer_format(text: str) -> bool:
    return bool(FORMAT_PATTERN.search(text))


# === Shared compute_metrics — logs all three metrics ===
def make_compute_metrics():
    """Returns a compute_metrics function that logs mean_reward, strict_accuracy, format_compliance."""

    def compute_metrics(eval_pred):
        rewards = eval_pred.label_ids  # (N, 1) gathered across GPUs
        completions = eval_pred.predictions  # list of completions
        answers = eval_pred.inputs  # list of ground truths (from your patch)

        mean_reward = rewards[:, 0].mean().item()

        strict_hits = sum(
            1 for c, gt in zip(completions, answers, strict=False) if strict_correct(gt, c[0]["content"])
        )
        strict_accuracy = strict_hits / len(completions) if completions else 0.0

        format_count = sum(1 for c in completions if has_answer_format(c[0]["content"]))
        format_compliance = format_count / len(completions) if completions else 0.0

        print("\n=== compute_metrics called ===")
        print(f"completions received: {len(completions)}")
        for i in range(min(3, len(completions))):
            print(f"  [{i}] expected={answers[i]} | text={completions[i][0]['content'][:80]}")
        print(f"mean_reward:       {mean_reward:.3f}")
        print(f"strict_accuracy:   {strict_accuracy:.3f}")
        print(f"format_compliance: {format_compliance:.3f}")

        return {
            "mean_reward": mean_reward,
            "strict_accuracy": strict_accuracy,
            "format_compliance": format_compliance,
        }

    return compute_metrics


# === Dataset prep ===
SYSTEM_PROMPT = (
    "Solve the math problem step by step. Wrap your final answer in <answer> tags, e.g. <answer>42</answer>."
)


def format_sample(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ],
        "answer": example["answer"].split("####")[-1].strip(),
    }
