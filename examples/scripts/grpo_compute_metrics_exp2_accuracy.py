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

# Experiment 2 of 4: accuracy reward only.
# Reward function: simple string correctness check.
# Held-out metrics: exact_match_simple, format_compliance, mean_reward.
# Baseline: Experiment 1 (format reward only) showed loss collapse at step 40
# with format_compliance=1.0 and exact_match=0.05.

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
        # Extract the numeric answer from GSM8K's "... #### 42" format
        "answer": example["answer"].split("####")[-1].strip(),
    }


def main():
    dataset = load_dataset("openai/gsm8k", "main")
    train_dataset = dataset["train"].map(format_sample)
    eval_dataset = dataset["test"].select(range(100)).map(format_sample)

    print("[Exp2] dataset sample | prompt:", train_dataset[0]["prompt"])
    print("[Exp2] dataset sample | answer:", train_dataset[0]["answer"])

    num_generations = 4

    # Capture eval answers before training starts. compute_metrics receives completions
    # but not dataset columns, so we close over the answers and index by position:
    # num_generations completions are produced per prompt, in dataset order.
    eval_answers = eval_dataset["answer"]

    # Shared helpers used by both accuracy_reward and compute_metrics to guarantee
    # identical extraction and matching logic across training and evaluation.

    def _extract_predicted(text: str) -> str:
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        return match.group(1).strip() if match else text

    def _answer_matches(gt: str, predicted: str) -> bool:
        gt = gt.strip()
        # Empty gt produces \b\b which matches everywhere — treat as no match.
        # Empty predicted has nothing to match against.
        if not gt or not predicted:
            return False
        # re.escape handles special regex chars in answers like "1.5" or "2/3".
        # \b prevents "4" from matching inside "14", "40", or "4.5".
        return bool(re.search(r"\b" + re.escape(gt) + r"\b", predicted))

    _reward_logged = [False]

    # Simple string matching: '42.0' and '42' may not match even if mathematically
    # equal. This is a known limitation. Experiment 3 will add format scaffolding
    # to help the model produce cleaner answers.
    def accuracy_reward(completions, answer, **kwargs):
        # Correctness reward: checks whether the ground truth answer appears in the
        # completion. Extracts from <answer> tags when present; otherwise searches
        # the full text. Returns 1.0 for correct, 0.0 for incorrect.
        # Does NOT use symbolic parsing — word-boundary string matching only.
        rewards = []
        for completion, gt in zip(completions, answer, strict=False):
            text = completion[0]["content"]
            predicted = _extract_predicted(text)
            rewards.append(1.0 if _answer_matches(gt, predicted) else 0.0)

        if not _reward_logged[0]:
            print("[Exp2] reward_func | completions received:", len(completions))
            print("[Exp2] reward_func | first completion text:", completions[0][0]["content"][:150])
            print("[Exp2] reward_func | first reward score:", rewards[0])
            _reward_logged[0] = True

        return rewards

    # At 0.5B scale, correctness reward may be sparse early in training. If most
    # completions score 0.0, group advantages collapse to zero and loss collapses.
    # This is expected behavior and a key finding about reward design at small scale.
    def compute_metrics(eval_pred):
        completions = eval_pred.predictions
        rewards = eval_pred.label_ids

        print("[Exp2] compute_metrics | completions received:", len(completions))
        print("[Exp2] compute_metrics | rewards shape:", rewards.shape)

        for i, completion in enumerate(completions[:3]):
            expected = eval_answers[i // num_generations] if i // num_generations < len(eval_answers) else "?"
            text = completion[0]["content"][:80]
            print(f"[Exp2] compute_metrics | [{i}] expected={expected!r} text={text!r}")

        # exact_match_simple: same extraction + word-boundary logic as accuracy_reward,
        # applied to the held-out eval set. Must stay in sync with accuracy_reward.
        exact_match_count = sum(
            1
            for i, completion in enumerate(completions)
            if i // num_generations < len(eval_answers)
            and _answer_matches(
                eval_answers[i // num_generations],
                _extract_predicted(completion[0]["content"]),
            )
        )
        exact_match_simple = exact_match_count / len(completions) if completions else 0.0

        # format_compliance: does the completion use <answer>...</answer> tags?
        # Kept from Experiment 1 for direct comparison.
        format_count = sum(
            1 for completion in completions if re.search(r"<answer>.*?</answer>", completion[0]["content"], re.DOTALL)
        )
        format_compliance = format_count / len(completions) if completions else 0.0

        # mean_reward: average of what accuracy_reward gave on eval completions.
        mean_reward = rewards[:, 0].mean().item()

        print(
            f"[Exp2] compute_metrics | exact_match_simple={exact_match_simple:.4f}"
            f" format_compliance={format_compliance:.4f}"
            f" mean_reward={mean_reward:.4f}"
        )

        return {
            "exact_match_simple": exact_match_simple,
            "format_compliance": format_compliance,
            "mean_reward": mean_reward,
        }

    trainer = GRPOTrainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        reward_funcs=accuracy_reward,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        args=GRPOConfig(
            num_generations=num_generations,
            eval_strategy="steps",
            eval_steps=50,
            max_steps=100,
            logging_steps=10,
            log_completions=True,
            per_device_train_batch_size=2,
            max_completion_length=512,
            report_to="none",
        ),
    )
    trainer.train()


if __name__ == "__main__":
    main()
