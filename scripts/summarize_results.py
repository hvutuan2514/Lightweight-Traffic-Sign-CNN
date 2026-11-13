"""Collect results/*_<split>_metrics.json into one comparison table.

Run from the repository root:
    python -m scripts.summarize_results            # validation results
    python -m scripts.summarize_results --split test

Prints a Markdown table and writes it to results/summary_<split>.md.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RESULTS_DIR

# Ablation variants first, then baselines.
ORDER = ["basic_cnn", "cnn_stn", "cnn_attention", "cnn_stn_attention", "resnet18", "hog_svm"]
LABELS = {
    "basic_cnn": "Basic CNN",
    "cnn_stn": "CNN + STN",
    "cnn_attention": "CNN + attention",
    "cnn_stn_attention": "CNN + STN + attention",
    "resnet18": "ResNet-18 (pretrained)",
    "hog_svm": "HOG + SVM",
}


def load_rows(split):
    rows = []
    for path in sorted(RESULTS_DIR.glob(f"*_{split}_metrics.json")):
        m = json.loads(path.read_text())
        rows.append(m)
    rows.sort(key=lambda m: (ORDER.index(m["run"]) if m["run"] in ORDER else len(ORDER), m["run"]))
    return rows


def to_markdown(rows, split):
    header = ("| Model | Parameters | Best epoch | Accuracy | Macro precision "
              "| Macro recall | Macro-F1 |")
    lines = [f"Results on the {split} split", "", header, "|---|---:|---:|---:|---:|---:|---:|"]
    for m in rows:
        params = f"{m['parameters']:,}" if "parameters" in m else "-"
        epoch = str(m.get("best_epoch", "-"))
        lines.append(f"| {LABELS.get(m['run'], m['run'])} | {params} | {epoch} "
                     f"| {m['accuracy']:.4f} | {m['macro_precision']:.4f} "
                     f"| {m['macro_recall']:.4f} | {m['macro_f1']:.4f} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=("val", "test"), default="val")
    args = p.parse_args(argv)

    rows = load_rows(args.split)
    if not rows:
        print(f"No results/*_{args.split}_metrics.json files found.")
        return
    table = to_markdown(rows, args.split)
    (RESULTS_DIR / f"summary_{args.split}.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
