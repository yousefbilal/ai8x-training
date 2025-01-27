import re
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import pandas as pd

parser = ArgumentParser()
parser.add_argument("file")

args = parser.parse_args()

log_file = Path(args.file)

if not log_file.exists():
    print("file does not exist")

with open(log_file) as f:
    text = f.read()

pattern = re.compile(
    r"test \(best\)[\s\S]*?/.+?/(.+?pth\.tar)[\s\S]*?(\[\[[\s\S]+?\]\])"
)

results = []

for m in pattern.finditer(text):
    ckpt = m.group(1)
    conf_matrix_str = m.group(2)
    lines = conf_matrix_str.strip().split("\n")

    cm = np.array([[int(x) for x in line.strip("[] ").split()] for line in lines])
    true_positive = np.diag(cm)
    false_positive = np.sum(cm, axis=0) - true_positive
    false_negative = np.sum(cm, axis=1) - true_positive
    true_negative = np.sum(cm) - (true_positive + false_positive + false_negative)

    # Compute metrics per class
    precision = true_positive / (true_positive + false_positive + 1e-9)
    recall = true_positive / (true_positive + false_negative + 1e-9)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
    accuracy = (true_positive + true_negative) / np.sum(cm)

    # Print per-class metrics
    for c in range(len(cm)):
        results.append(
            {
                "Checkpoint": ckpt,
                "Class": c,
                "Precision": precision[c],
                "Recall": recall[c],
                "F1 Score": f1[c],
                "Accuracy": accuracy[c],
            }
        )

    # Compute averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    overall_accuracy = np.sum(true_positive) / np.sum(cm)

    results.append(
        {
            "Checkpoint": ckpt,
            "Class": "Average",
            "Precision": macro_precision,
            "Recall": macro_recall,
            "F1 Score": macro_f1,
            "Accuracy": overall_accuracy,
        }
    )

csv_file = "classification_results.csv"
df = pd.DataFrame(results)
df.to_csv(csv_file, index=False)

# Export to Excel
excel_file = "classification_results.xlsx"
df.to_excel(excel_file, index=False)

print(f"Results exported to {csv_file} and {excel_file}.")
