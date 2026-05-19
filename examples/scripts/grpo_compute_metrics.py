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


def format_reward(completions, **kwargs):
    # Measures format compliance only — whether the completion uses <answer> tags.
    # The value inside the tags is NOT checked: a completion like <answer>wrong</answer>
    # scores 1.0 here. compute_metrics is the only signal that can catch this.
    return [
        1.0 if re.search(r"<answer>.*?</answer>", completion[0]["content"], re.DOTALL) else 0.0
        for completion in completions
    ]


def main():
    dataset = load_dataset("openai/gsm8k", "main")
    train_dataset = dataset["train"].map(format_sample)
    eval_dataset = dataset["test"].select(range(100)).map(format_sample)

    num_generations = 4

    # Capture eval answers before training starts. compute_metrics receives completions
    # but not dataset columns, so we close over the answers and index by position:
    # num_generations completions are produced per prompt, in dataset order.
    eval_answers = eval_dataset["answer"]

    def compute_metrics(eval_pred):
        # Measures correctness only — completely independent of the format reward.
        # eval_pred.label_ids holds format compliance scores, not ground truth answers.
        # If eval_reward climbs while eval_exact_match stays flat, the model is
        # gaming the format signal rather than learning to solve math problems.
        completions = eval_pred.predictions
        correct = sum(
            eval_answers[i // num_generations] in completion[0]["content"]
            for i, completion in enumerate(completions)
            if i // num_generations < len(eval_answers)
        )
        return {"exact_match": correct / len(completions) if completions else 0.0}

    trainer = GRPOTrainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        reward_funcs=format_reward,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        args=GRPOConfig(
            num_generations=num_generations,
            eval_strategy="steps",
            eval_steps=50,
            logging_steps=10,
            log_completions=True,
            per_device_train_batch_size=2,
            max_completion_length=512,
        ),
    )
    trainer.train()


if __name__ == "__main__":
    main()
