"""
Convert the 50-completion validation sample into a CSV you can hand-label.

You'll fill in the 'my_label' column manually:
  - Write the number the model intended as its final answer (e.g. "26", "8")
  - Write "NONE" if the completion is truncated/incoherent with no discernible answer

Don't look at the judge's output while labeling — that would defeat the purpose.

Usage:
    python build_labeling_csv.py --input judge_validation_sample.json \
                                  --output validation_to_label.csv
"""

import argparse
import csv
import json


def build_csv(input_path: str, output_path: str):
    with open(input_path) as f:
        items = json.load(f)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "problem", "gt", "completion", "my_label"])
        for i, item in enumerate(items):
            writer.writerow(
                [
                    i,
                    item["problem"],
                    item["gt"],
                    item["completion"],
                    "",  # blank — you fill this in
                ]
            )

    print(f"Wrote {len(items)} rows to {output_path}")
    print("Open in a spreadsheet, fill in 'my_label' column with the model's intended answer.")
    print("Leave blank or write 'NONE' if the completion has no discernible final answer.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="judge_validation_sample.json")
    parser.add_argument("--output", default="validation_to_label.csv")
    args = parser.parse_args()
    build_csv(args.input, args.output)
