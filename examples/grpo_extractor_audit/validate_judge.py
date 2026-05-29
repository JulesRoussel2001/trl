"""
Validate the judge against your hand-labels.

Reads:
  - validation_labeled.csv (your hand-labels in the 'my_label' column)
  - judge_validation_sample.json (the original 50 completions)

Runs the judge on the same 50 completions, then computes agreement.

Output:
  - judge_validation_results.json (per-completion: your label vs judge label)
  - prints agreement % and lists disagreements for inspection

Usage:
    python validate_judge.py --labeled validation_labeled.csv \
                              --sample judge_validation_sample.json
"""

import argparse
import csv
import json
import os

from anthropic import Anthropic

# Import the judge call from audit_extractors.py
# (Assumes both files are in the same directory.)
from audit_extractors import judge_completion, normalize_answer


def normalize_label(s: str) -> str:
    """Normalize labels so '8.0' and '8' compare equal, etc."""
    if not s or s.strip().upper() == "NONE":
        return "NONE"
    try:
        return normalize_answer(s)
    except Exception:
        return s.strip()


def load_human_labels(csv_path: str) -> dict[int, str]:
    """Read the hand-labeled CSV; returns {idx: normalized_label}."""
    labels = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[int(row["idx"])] = normalize_label(row["my_label"])
    return labels


def run_validation(
    labeled_csv: str,
    sample_json: str,
    output_path: str = "judge_validation_results.json",
    judge_model: str = "claude-haiku-4-5-20251001",
):
    human_labels = load_human_labels(labeled_csv)
    with open(sample_json) as f:
        sample = json.load(f)

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    results = []
    for i, item in enumerate(sample):
        judgment = judge_completion(
            client,
            item["problem"],
            item["gt"],
            item["completion"],
            model=judge_model,
        )
        my_label = human_labels.get(i, "")
        judge_label = normalize_label(judgment.get("intended_answer", ""))
        agree = my_label == judge_label

        results.append(
            {
                "idx": i,
                "my_label": my_label,
                "judge_label": judge_label,
                "agree": agree,
                "judge_reasoning": judgment.get("reasoning", ""),
                "completion": item["completion"],
            }
        )

    # Compute and report agreement
    n_total = len(results)
    n_agree = sum(1 for r in results if r["agree"])
    agreement = n_agree / n_total if n_total else 0.0

    print("\n=== Judge validation ===")
    print(f"Total: {n_total}")
    print(f"Agreement: {n_agree}/{n_total} = {agreement:.1%}")

    disagreements = [r for r in results if not r["agree"]]
    if disagreements:
        print(f"\n--- {len(disagreements)} disagreements (worth inspecting) ---")
        for r in disagreements:
            print(f"\nidx={r['idx']}")
            print(f"  my label:    {r['my_label']}")
            print(f"  judge label: {r['judge_label']}")
            print(f"  judge says:  {r['judge_reasoning']}")
            print(f"  completion (last 200 chars): ...{r['completion'][-200:]}")

    with open(output_path, "w") as f:
        json.dump(
            {
                "agreement": agreement,
                "n_total": n_total,
                "n_agree": n_agree,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nFull results written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labeled", default="validation_labeled.csv", help="CSV with your hand-labels in 'my_label' column"
    )
    parser.add_argument("--sample", default="judge_validation_sample.json", help="The 50-completion sample JSON")
    parser.add_argument("--output", default="judge_validation_results.json")
    parser.add_argument("--judge-model", default="claude-haiku-4-5-20251001")
    args = parser.parse_args()

    run_validation(
        labeled_csv=args.labeled,
        sample_json=args.sample,
        output_path=args.output,
        judge_model=args.judge_model,
    )
