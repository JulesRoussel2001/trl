import csv


with open("validation_to_label.csv") as f:
    blanks = [row["idx"] for row in csv.DictReader(f) if not row["my_label"].strip()]

if blanks:
    print("Unlabeled rows:", blanks)
else:
    print("All rows labeled ✓")
