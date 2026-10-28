"""Measure the trained model on the test boards it never saw.

    cv-evaluate [--model models/pcb_nano.pth] [--split test] [--threshold 0.4]

Two views: box mAP (does it find each defect where it is?) and the one quality control
cares about, per board (does it reject the defective boards, and pass the good ones?).
Writes the numbers to <model>.metrics.json.
"""

import argparse
import json
import os

import numpy as np
from PIL import Image

from .detector import DEFAULT_MODEL, Detector


def load_split(dataset, split):
    """``[(path, [(label, (x0, y0, x1, y1))])]`` of a split, boxes in pixels."""
    with open(os.path.join(dataset, split, "_annotations.coco.json")) as handle:
        coco = json.load(handle)
    names = {c["id"]: c["name"] for c in coco["categories"]}
    boxes = {image["id"]: [] for image in coco["images"]}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        boxes[a["image_id"]].append((names[a["category_id"]], (x, y, x + w, y + h)))
    return [(os.path.join(dataset, split, image["file_name"]), boxes[image["id"]]) for image in coco["images"]]


def board_metrics(truths, predictions):
    """Per board: ``truths`` and ``predictions`` are lists of booleans (defective or not)."""
    tp = sum(t and p for t, p in zip(truths, predictions))
    fp = sum(p and not t for t, p in zip(truths, predictions))
    fn = sum(t and not p for t, p in zip(truths, predictions))
    tn = sum(not t and not p for t, p in zip(truths, predictions))
    return {
        "boards": len(truths),
        "defective_caught": tp,
        "defective_missed": fn,
        "good_passed": tn,
        "good_rejected": fp,
        "precision": round(tp / (tp + fp), 3) if tp + fp else None,
        "recall": round(tp / (tp + fn), 3) if tp + fn else None,
        "accuracy": round((tp + tn) / len(truths), 3) if truths else None,
    }


def evaluate(model=DEFAULT_MODEL, dataset="dataset", split="test", threshold=0.4):
    import supervision as sv
    from supervision.metrics import MeanAveragePrecision

    detector = Detector(model, threshold=0.01)
    names = list(detector.names)
    metric = MeanAveragePrecision()
    truths, predictions = [], []
    for path, boxes in load_split(dataset, split):
        image = Image.open(path)
        found = detector.detect(image)
        width, height = image.size
        metric.update(
            sv.Detections(
                xyxy=np.array([[d["bbox"]["x"] * width, d["bbox"]["y"] * height,
                                (d["bbox"]["x"] + d["bbox"]["w"]) * width, (d["bbox"]["y"] + d["bbox"]["h"]) * height]
                               for d in found], dtype=float).reshape(-1, 4),
                class_id=np.array([names.index(d["label"]) for d in found], dtype=int),
                confidence=np.array([d["confidence"] for d in found], dtype=float),
            ),
            sv.Detections(
                xyxy=np.array([box for _label, box in boxes], dtype=float).reshape(-1, 4),
                class_id=np.array([names.index(label) for label, _box in boxes], dtype=int),
            ),
        )
        truths.append(bool(boxes))
        predictions.append(any(d["confidence"] >= threshold for d in found))
    result = metric.compute()
    metrics = {
        "model": os.path.basename(model),
        "split": split,
        "threshold": threshold,
        "map50": round(float(result.map50), 3),
        "map50_95": round(float(result.map50_95), 3),
        "boards": board_metrics(truths, predictions),
    }
    with open(os.path.splitext(model)[0] + ".metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--split", default="test", choices=("train", "valid", "test"))
    parser.add_argument("--threshold", type=float, default=0.4)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.model, args.dataset, args.split, args.threshold), indent=2))


if __name__ == "__main__":
    main()
