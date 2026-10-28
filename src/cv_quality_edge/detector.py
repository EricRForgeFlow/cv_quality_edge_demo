"""Run the trained model on board images.

    cv-detect IMAGE [IMAGE ...] [--model models/pcb_nano.pth] [--threshold 0.4]

Prints the defects found and writes the images with their boxes to runs/detect/.
"""

import argparse
import os
import time

from PIL import Image, ImageDraw, ImageFont

DEFAULT_MODEL = "models/pcb_nano.pth"
COLORS = {"bent": (240, 200, 40), "melt": (245, 140, 40), "scratch": (60, 180, 245), "missing": (230, 60, 60)}


class Detector:
    """A trained RF-DETR checkpoint, detecting on CPU (or a GPU when there is one)."""

    def __init__(self, checkpoint=DEFAULT_MODEL, threshold=0.4):
        from rfdetr import RFDETR

        self.model = RFDETR.from_checkpoint(checkpoint)
        self.names = self.model.class_names
        self.threshold = threshold
        self.version = os.path.splitext(os.path.basename(checkpoint))[0]

    def detect(self, image):
        """The defects of a PIL image: ``[{label, confidence, bbox: {x, y, w, h}}]``, boxes relative to the image."""
        detections = self.model.predict(image.convert("RGB"), threshold=self.threshold)
        width, height = image.size
        found = []
        labels = detections.data.get("class_name")
        if labels is None:
            labels = [self.names[int(class_id)] for class_id in detections.class_id]
        for (x0, y0, x1, y1), label, confidence in zip(detections.xyxy, labels, detections.confidence):
            if label not in self.names:  # the "no object" slot, only kept at very low thresholds
                continue
            found.append({
                "label": str(label),
                "confidence": round(float(confidence), 3),
                "bbox": {
                    "x": round(max(0.0, float(x0)) / width, 4),
                    "y": round(max(0.0, float(y0)) / height, 4),
                    "w": round((min(float(x1), width) - max(0.0, float(x0))) / width, 4),
                    "h": round((min(float(y1), height) - max(0.0, float(y0))) / height, 4),
                },
            })
        return sorted(found, key=lambda d: -d["confidence"])


def annotate(image, detections):
    """A copy of ``image`` with the boxes and labels of ``detections`` drawn on it."""
    image = image.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    width, height = image.size
    line = max(2, round(width / 400))
    font = ImageFont.load_default(size=max(12, round(width / 60)))
    for detection in detections:
        box = detection["bbox"]
        x0, y0 = box["x"] * width, box["y"] * height
        x1, y1 = x0 + box["w"] * width, y0 + box["h"] * height
        color = COLORS.get(detection["label"], (255, 255, 255))
        draw.rectangle([x0, y0, x1, y1], outline=color, width=line)
        text = f"{detection['label']} {round(detection['confidence'] * 100)}%"
        left, top, right, bottom = draw.textbbox((x0, y0), text, font=font)
        label_top = max(0, y0 - (bottom - top) - 4)
        draw.rectangle([x0, label_top, x0 + (right - left) + 6, label_top + (bottom - top) + 4], fill=color)
        draw.text((x0 + 3, label_top + 1), text, fill=(0, 0, 0), font=font)
    return image


def describe(detections):
    return ", ".join(f"{d['label']} {d['confidence']:.2f}" for d in detections) or "nothing"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("images", nargs="+")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--out", default="runs/detect")
    args = parser.parse_args()
    detector = Detector(args.model, args.threshold)
    os.makedirs(args.out, exist_ok=True)
    for path in args.images:
        image = Image.open(path)
        start = time.perf_counter()
        detections = detector.detect(image)
        elapsed = (time.perf_counter() - start) * 1000
        annotate(image, detections).save(os.path.join(args.out, os.path.basename(path)))
        print(f"{os.path.basename(path)}: {describe(detections)} ({elapsed:.0f} ms)")


if __name__ == "__main__":
    main()
