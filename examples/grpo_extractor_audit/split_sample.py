"""
Split the generated audit sample into:
  - judge_validation_sample.json (50 completions for hand-labeling + judge agreement check)
  - audit_sample.json (remaining ~500 completions for the actual extractor audit)

The split is random with a fixed seed for reproducibility.

Usage:
    python split_sample.py --input audit_sample_full.json --n-validation 50
"""

import argparse
import json
import random


def split_sample(
    input_path: str,
    validation_path: str = "judge_validation_sample.json",
    audit_path: str = "audit_sample.json",
    n_validation: int = 50,
    seed: int = 42,
):
    with open(input_path) as f:
        all_completions = json.load(f)

    if len(all_completions) < n_validation + 50:
        print(f"WARNING: only {len(all_completions)} completions available; recommend at least {n_validation + 500}.")

    random.seed(seed)
    indices = list(range(len(all_completions)))
    random.shuffle(indices)

    validation_idx = indices[:n_validation]
    audit_idx = indices[n_validation:]

    validation_set = [all_completions[i] for i in validation_idx]
    audit_set = [all_completions[i] for i in audit_idx]

    with open(validation_path, "w") as f:
        json.dump(validation_set, f, indent=2)
    with open(audit_path, "w") as f:
        json.dump(audit_set, f, indent=2)

    print(f"Validation set: {len(validation_set)} → {validation_path}")
    print(f"Audit set:      {len(audit_set)} → {audit_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="audit_sample_full.json")
    parser.add_argument("--validation-output", default="judge_validation_sample.json")
    parser.add_argument("--audit-output", default="audit_sample.json")
    parser.add_argument("--n-validation", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_sample(
        input_path=args.input,
        validation_path=args.validation_output,
        audit_path=args.audit_output,
        n_validation=args.n_validation,
        seed=args.seed,
    )
