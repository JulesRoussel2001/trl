"""Local smoke-test for grpo_compute_metrics.py.

Uses a tiny model and a handful of GSM8K examples to verify the full pipeline
(reward function → buffer accumulation → compute_metrics) works end to end.
Not intended to train a useful model.
"""

import re

from datasets import load_dataset

from trl import GRPOConfig, GRPOTrainer


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


# ── Debug flag for format_reward ──────────────────────────────────────────────
_format_reward_first_call = [True]


def format_reward(completions, **kwargs):
    # Measures format compliance only — the value inside <answer> is NOT checked.
    rewards = [
        1.0 if re.search(r"<answer>.*?</answer>", completion[0]["content"], re.DOTALL) else 0.0
        for completion in completions
    ]

    if _format_reward_first_call[0]:
        _format_reward_first_call[0] = False
        first_content = completions[0][0]["content"]
        print("\n── format_reward (first call) ─────────────────────────────────────")
        print(f"  completions received : {len(completions)}")
        print(f"  first completion text: {repr(first_content)}")
        print(f"  reward assigned      : {rewards[0]}")
        print("───────────────────────────────────────────────────────────────────\n")

    return rewards


def main():
    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = load_dataset("openai/gsm8k", "main")
    train_dataset = dataset["train"].select(range(5)).map(format_sample)
    eval_dataset = dataset["test"].select(range(5)).map(format_sample)

    print("\n── Dataset samples ────────────────────────────────────────────────")
    train_ex = train_dataset[0]
    print(f"  train[0] prompt : {train_ex['prompt']}")
    print(f"  train[0] answer : {train_ex['answer']}")
    eval_ex = eval_dataset[0]
    print(f"  eval[0]  prompt : {eval_ex['prompt']}")
    print(f"  eval[0]  answer : {eval_ex['answer']}")
    print("───────────────────────────────────────────────────────────────────\n")

    num_generations = 2
    eval_answers = eval_dataset["answer"]

    print("\n── eval_answers (first 5) ─────────────────────────────────────────")
    for i, ans in enumerate(eval_answers[:5]):
        print(f"  eval_answers[{i}] = {repr(ans)}")
    print("───────────────────────────────────────────────────────────────────\n")

    def compute_metrics(eval_pred):
        completions = eval_pred.predictions
        rewards_tensor = eval_pred.label_ids

        print("\n── compute_metrics ────────────────────────────────────────────────")
        print(f"  total completions received : {len(completions)}")
        print(f"  rewards tensor shape       : {rewards_tensor.shape}")
        for i in range(min(3, len(completions))):
            answer_idx = i // num_generations
            content = completions[i][0]["content"]
            mapped_answer = eval_answers[answer_idx] if answer_idx < len(eval_answers) else "OUT_OF_RANGE"
            print(f"  completions[{i}] (→ eval_answers[{answer_idx}]={repr(mapped_answer)}): {repr(content)}")

        correct = sum(
            eval_answers[i // num_generations] in completion[0]["content"]
            for i, completion in enumerate(completions)
            if i // num_generations < len(eval_answers)
        )
        exact_match = correct / len(completions) if completions else 0.0
        format_compliance = rewards_tensor[:, 0].mean().item()

        print(f"  eval_format_compliance     : {format_compliance:.4f}")
        print(f"  eval_exact_match           : {exact_match:.4f}")
        print("───────────────────────────────────────────────────────────────────\n")

        return {"format_compliance": format_compliance, "exact_match": exact_match}

    # ── Training ──────────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
        reward_funcs=format_reward,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        args=GRPOConfig(
            num_generations=num_generations,
            max_steps=2,
            eval_strategy="steps",
            eval_steps=1,
            logging_steps=1,
            log_completions=False,
            per_device_train_batch_size=2,
            max_completion_length=8,
        ),
    )
    trainer.train()


if __name__ == "__main__":
    main()
