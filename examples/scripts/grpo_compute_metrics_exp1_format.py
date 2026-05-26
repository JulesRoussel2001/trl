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
Experiment 1 — Format reward only.

Hypothesis: rewarding only format (the presence of <answer> tags) leads
to reward hacking — format compliance goes to 1.0 while actual task
performance degrades.

Reward:    format-only (1.0 if <answer>...</answer> present, else 0.0)
Observed:  mean_reward (= format_compliance), strict_accuracy, format_compliance
"""

from datasets import load_dataset
from grpo_compute_metrics_utils import (
    format_sample,
    has_answer_format,
    make_compute_metrics,
)

from trl import GRPOConfig, GRPOTrainer


def make_format_reward():
    """Returns a fresh format_reward function with its own first-call flag.

    Why a factory: the 'first call' debug print uses a flag in the closure.
    If we defined first_call at module level, it would stay False after the
    first run, so re-running the notebook wouldn't show the debug print.
    Calling make_format_reward() each time gives us a brand-new flag.
    """
    first_call = [True]

    def format_reward(completions, answer, **kwargs):
        """1.0 if the completion contains <answer>...</answer>, else 0.0."""
        rewards = [1.0 if has_answer_format(c[0]["content"]) else 0.0 for c in completions]
        if first_call[0]:
            print("\n=== First reward_func call ===")
            print(f"completions received: {len(completions)}")
            print(f"first completion text: {completions[0][0]['content'][:150]}")
            print(f"first reward score: {rewards[0]}")
            first_call[0] = False
        return rewards

    return format_reward


def main():
    dataset = load_dataset("openai/gsm8k", "main")
    train_dataset = dataset["train"].select(range(200)).map(format_sample)
    eval_dataset = dataset["test"].select(range(50)).map(format_sample)

    print("=== Dataset sample ===")
    print(f"train example prompt: {train_dataset[0]['prompt']}")
    print(f"train example answer: {train_dataset[0]['answer']}")
    print(f"eval answers (first 5): {eval_dataset['answer'][:5]}")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        reward_funcs=make_format_reward(),
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


if __name__ == "__main__":
    main()
