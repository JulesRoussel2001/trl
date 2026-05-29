"""
Audit extractor precision/recall against an LLM judge.

What this does:
1. Loads N completions sampled across your training runs (mix of experiments and steps)
2. For each completion, asks Claude Opus 4.7 to identify (a) the model's intended
   numerical answer, and (b) whether that answer matches the ground truth
3. Compares each extractor's output (lenient substring, last-number, strict-tag)
   against the judge's verdict
4. Produces a precision/recall/F1 table per extractor

Output: audit_results.json with per-completion judgments and aggregate metrics.

Cost estimate: ~500 completions × 1 call each ≈ $2-5 at Opus pricing.
"""

import json
import os
import re
import time

from anthropic import Anthropic
from dotenv import load_dotenv


load_dotenv()


# === Local imports: your extractors and matchers ===
# Copy these from eval_utils.py so the audit script is standalone
def normalize_answer(s: str) -> str:
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    s = s.replace(" ", "")
    s = s.lstrip("+")
    try:
        num = float(s)
        s = str(int(num)) if num == int(num) else str(num)
    except ValueError:
        pass
    return s


def answer_in_text_lenient(gt: str, text: str) -> bool:
    """Lenient substring matcher (Exp 1/2/3 reward style)."""
    gt = normalize_answer(gt)
    if not gt or not text:
        return False
    text_normalized = text.replace(",", "").replace("$", "").replace("%", "")
    pattern = r"(?<![\d.-])" + re.escape(gt) + r"(?!\d|\.[\d])"
    return re.search(pattern, text_normalized) is not None


def extract_last_number(text: str) -> str | None:
    """Tiered: <answer> → \\boxed → last number in text."""

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


def extract_strict_tag(text: str) -> str | None:
    """Only count answer if inside <answer>...</answer> tags."""
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if not m:
        return None
    nums = re.findall(r"[\d,]+(?:\.\d+)?", m.group(1))
    return nums[-1].replace(",", "") if nums else None


def correct_via_extractor(extractor_fn, gt: str, text: str) -> bool:
    """Use any extractor to determine correctness via numeric equality."""
    pred = extractor_fn(text)
    if pred is None:
        return False
    try:
        return float(pred) == float(normalize_answer(gt))
    except ValueError:
        return False


# === Judge call via Anthropic API with tool use for structured output ===

JUDGE_TOOL = {
    "name": "record_judgment",
    "description": "Record the judge's assessment of a model completion.",
    "input_schema": {
        "type": "object",
        "properties": {
            "intended_answer": {
                "type": "string",
                "description": "The single numerical value the model intends as its final answer. "
                "Strip currency signs, units, and commas. If the model produced no "
                "discernible final answer, return 'NONE'.",
            },
            "is_correct": {
                "type": "boolean",
                "description": "Whether the intended answer matches the ground truth numerically. "
                "If intended_answer is 'NONE', return false.",
            },
            "reasoning": {
                "type": "string",
                "description": "Brief (1-2 sentences) explanation of how you identified the intended "
                "answer. Mention if the completion was truncated or incoherent.",
            },
        },
        "required": ["intended_answer", "is_correct", "reasoning"],
    },
}

JUDGE_SYSTEM_PROMPT = """You are evaluating a math problem completion to identify what numerical \
answer the model actually intended as its final answer, and whether that answer is correct.

Identify the model's intended final answer:
- If the model wrote a final answer in <answer>...</answer> tags, use that.
- If the model wrote a final answer in \\boxed{...}, use that.
- Otherwise, identify the final answer from the prose (e.g., "Therefore, the answer is 26").
- If the completion is incoherent or truncated before reaching an answer, return 'NONE'.

Do not assume the answer is correct just because it appears in the reasoning. The model often \
mentions intermediate numbers that aren't its final answer.

Then check if that intended answer equals the ground truth numerically (ignoring units, $, %, commas)."""


def judge_completion(
    client: Anthropic,
    problem: str,
    gt: str,
    completion: str,
    model: str = "claude-haiku-4-5-20251001",
    max_retries: int = 3,
) -> dict:
    """Single judge call. Returns dict with intended_answer, is_correct, reasoning."""
    user_msg = f"""Problem: {problem}

Ground truth answer: {gt}

Model completion:
---
{completion}
---

Identify the model's intended final answer and whether it matches the ground truth."""

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=500,
                tools=[JUDGE_TOOL],
                tool_choice={"type": "tool", "name": "record_judgment"},
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            # Extract the tool use block
            for block in response.content:
                if block.type == "tool_use" and block.name == "record_judgment":
                    return block.input
            raise ValueError("No tool_use block in response")
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2**attempt)  # exponential backoff
                continue
            return {
                "intended_answer": "ERROR",
                "is_correct": False,
                "reasoning": f"Judge call failed after {max_retries} retries: {e}",
            }


# === Main audit loop ===


def run_audit(
    completions: list[dict],
    output_path: str,
    api_key: str | None = None,
    judge_model: str = "claude-haiku-4-5-20251001",
):
    """Run the audit and write results."""
    client = Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    results = []
    for i, item in enumerate(completions):
        if i % 25 == 0:
            print(f"[{i}/{len(completions)}] judging...")

        judgment = judge_completion(client, item["problem"], item["gt"], item["completion"], model=judge_model)

        # Run each extractor
        lenient_correct = answer_in_text_lenient(item["gt"], item["completion"])
        last_num_correct = correct_via_extractor(extract_last_number, item["gt"], item["completion"])
        strict_correct = correct_via_extractor(extract_strict_tag, item["gt"], item["completion"])

        results.append(
            {
                "idx": i,
                "exp": item.get("exp"),
                "step": item.get("step"),
                "problem": item["problem"][:200],  # truncate for storage
                "gt": item["gt"],
                "completion": item["completion"],
                "judge": judgment,
                "lenient_correct": lenient_correct,
                "last_num_correct": last_num_correct,
                "strict_correct": strict_correct,
            }
        )

        # Write incrementally so we don't lose progress on a crash
        if i % 25 == 0:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def compute_metrics(results: list[dict]):
    """Compute precision/recall/F1 per extractor against judge ground truth."""
    # Filter to cases where judge gave a verdict (skip ERROR rows)
    valid = [r for r in results if r["judge"]["intended_answer"] != "ERROR"]

    def pr(extractor_key: str):
        # judge_correct = ground truth label (true if model's intended answer was correct)
        # extractor_correct = what the extractor said
        tp = sum(1 for r in valid if r[extractor_key] and r["judge"]["is_correct"])
        fp = sum(1 for r in valid if r[extractor_key] and not r["judge"]["is_correct"])
        fn = sum(1 for r in valid if not r[extractor_key] and r["judge"]["is_correct"])
        tn = sum(1 for r in valid if not r[extractor_key] and not r["judge"]["is_correct"])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}

    return {
        "n_valid": len(valid),
        "n_total": len(results),
        "lenient": pr("lenient_correct"),
        "last_num": pr("last_num_correct"),
        "strict": pr("strict_correct"),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="audit_sample.json", help="JSON file of completions to audit")
    parser.add_argument("--output", default="audit_results.json", help="Where to write per-completion judgments")
    parser.add_argument("--metrics-output", default="audit_metrics.json", help="Where to write aggregate metrics")
    parser.add_argument(
        "--judge-model", default="claude-haiku-4-5-20251001", help="Anthropic model to use as the judge"
    )
    args = parser.parse_args()

    with open(args.input) as f:
        completions = json.load(f)

    results = run_audit(completions, output_path=args.output, judge_model=args.judge_model)
    metrics = compute_metrics(results)

    print("\n=== Audit metrics ===")
    print(f"Valid judgments: {metrics['n_valid']}/{metrics['n_total']}")
    for name in ["lenient", "last_num", "strict"]:
        m = metrics[name]
        print(f"\n{name}:")
        print(f"  precision: {m['precision']:.3f}")
        print(f"  recall:    {m['recall']:.3f}")
        print(f"  F1:        {m['f1']:.3f}")
        print(f"  TP/FP/FN/TN: {m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}")

    with open(args.metrics_output, "w") as f:
        json.dump(metrics, f, indent=2)
