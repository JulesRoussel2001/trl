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

"""
Experiment 3 — Combined reward: 0.3 × format + 0.7 × correctness.

Hypothesis: combining a dense format signal with a sparse correctness
signal avoids both failure modes — format provides early learning
gradient, correctness prevents reward hacking. The weighting (0.3/0.7)
ensures correctness dominates, so format alone cannot maximize reward.

Reward:    0.3 * has_format + 0.7 * is_correct (lenient substring)
Observed:  mean_reward, strict_accuracy, format_compliance
"""

from datasets import load_dataset
from eval_utils import (
    answer_in_text,
    format_sample,
    has_answer_format,
    make_compute_metrics,
)

from trl import GRPOConfig, GRPOTrainer


# === Data ===
dataset = load_dataset("openai/gsm8k", "main")
train_dataset = dataset["train"].select(range(200)).map(format_sample)
eval_dataset = dataset["test"].select(range(50)).map(format_sample)

print("=== Dataset sample ===")
print(f"train example prompt: {train_dataset[0]['prompt']}")
print(f"train example answer: {train_dataset[0]['answer']}")
print(f"eval answers (first 5): {eval_dataset['answer'][:5]}")


# === Reward function factory: combined (0.3 format + 0.7 correctness) ===
def make_combined_reward():
    """Returns a fresh combined_reward function with its own first-call flag."""
    first_call = [True]

    def combined_reward(completions, answer, **kwargs):
        """0.3 * has_format + 0.7 * is_correct.

        Format is necessary-but-not-sufficient for max reward. Model must
        learn correctness to score above group average, preventing the
        pure format-hacking collapse seen in Exp 1.
        """
        rewards = []
        for c, gt in zip(completions, answer, strict=False):
            text = c[0]["content"]
            has_format = 1.0 if has_answer_format(text) else 0.0
            is_correct = 1.0 if answer_in_text(gt, text) else 0.0
            rewards.append(0.3 * has_format + 0.7 * is_correct)

        if first_call[0]:
            print("\n=== First reward_func call ===")
            print(f"completions received: {len(completions)}")
            print(f"first completion text: {completions[0][0]['content'][:150]}")
            print(f"first reward score: {rewards[0]}")
            first_call[0] = False
        return rewards

    return combined_reward


# === Trainer ===
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=make_combined_reward(),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=make_compute_metrics(),
    args=GRPOConfig(
        eval_on_start=True,
        num_generations=8,
        eval_strategy="steps",
        eval_steps=20,
        logging_steps=5,
        max_steps=100,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        max_completion_length=512,
        report_to="none",
    ),
)

trainer.train()
