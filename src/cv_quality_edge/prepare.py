"""VisA circuit boards → an object detection dataset in COCO format, as RF-DETR trains on.

    cv-prepare                       # download VisA if needed, then build dataset/
    cv-prepare --visa data/visa      # from an extracted copy

VisA marks defects with segmentation masks; each connected region of a class becomes a
box. Good boards go in as images without boxes, so the model also learns to find nothing.
"""

import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np
from PIL import Image
from scipy import ndimage

from . import visa

MIN_AREA = 30  # pixels of a mask region, at full resolution: smaller ones are noise
MERGE_GAP = 6  # regions of a class closer than this (pixels) are one defect
SPLITS = {"train": 0.70, "valid": 0.15, "test": 0.15}


def mask_boxes(mask, min_area=MIN_AREA, gap=MERGE_GAP):
    """The defects of a multi-class mask: ``[(class_id, (x0, y0, x1, y1))]``, in pixels.

    One box per connected region of each class; regions closer than ``gap`` are merged, so a
    scratch drawn in several strokes stays one box.
    """
    boxes = []
    for class_id in (int(v) for v in np.unique(mask) if v):
        pixels = mask == class_id
        grown = ndimage.binary_dilation(pixels, iterations=gap) if gap else pixels
        labels, _count = ndimage.label(grown)
        for index, window in enumerate(ndimage.find_objects(labels), start=1):
            ys, xs = np.nonzero((labels[window] == index) & pixels[window])
            if len(xs) < min_area:
                continue
            top, left = window[0].start, window[1].start
            boxes.append((class_id, (left + int(xs.min()), top + int(ys.min()),
                                     left + int(xs.max()) + 1, top + int(ys.max()) + 1)))
    return boxes


def split(items, key, seed, ratios=SPLITS):
    """Assign each item a split, stratified by ``key(item)``: ``{split: [items]}``.

    Every group of three or more items has at least one item in each split.
    """
    groups = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    rng = random.Random(seed)
    result = {name: [] for name in ratios}
    for group_key in sorted(groups):
        group = sorted(groups[group_key], key=lambda item: item["image"])
        rng.shuffle(group)
        n_valid = max(1, round(len(group) * ratios["valid"])) if len(group) >= 3 else 0
        n_test = max(1, round(len(group) * ratios["test"])) if len(group) >= 3 else 0
        result["valid"] += group[:n_valid]
        result["test"] += group[n_valid:n_valid + n_test]
        result["train"] += group[n_valid + n_test:]
    return result


def read_object(root, name):
    """The images of a VisA object: ``[{object, image, mask, label}]``, paths absolute."""
    folder = os.path.join(root, name)
    with open(os.path.join(folder, "image_anno.csv"), newline="") as handle:
        rows = list(csv.DictReader(handle))
    base = os.path.dirname(folder)
    items = []
    for row in rows:
        items.append({
            "object": name,
            "image": os.path.join(base, row["image"]),
            "mask": os.path.join(base, row["mask"]) if row.get("mask") else None,
            "label": row["label"],
        })
    return items


def annotate(item):
    """Read the mask of an anomalous item: its boxes (full resolution) and dominant class."""
    mask = np.array(Image.open(item["mask"]))
    if mask.ndim == 3:
        mask = mask[..., 0]
    values = set(np.unique(mask).tolist()) - {0}
    if values - set(visa.CLASSES):
        raise ValueError(f"{item['mask']}: mask values {sorted(values)} aren't the classes {visa.CLASSES}")
    boxes = mask_boxes(mask)
    areas = Counter()
    for class_id, (x0, y0, x1, y1) in boxes:
        areas[class_id] += (x1 - x0) * (y1 - y0)
    return boxes, areas.most_common(1)[0][0] if areas else None


def build(root, out, objects=visa.OBJECTS, normal_ratio=1.0, max_side=1024, seed=42):
    """Write the COCO dataset to ``out``; returns its summary."""
    anomalous, normal, skipped = [], [], []
    for name in objects:
        for item in read_object(root, name):
            if item["label"] == "normal":
                normal.append(item)
                continue
            item["boxes"], item["class"] = annotate(item)
            (anomalous if item["boxes"] else skipped).append(item)
    splits = split(anomalous, key=lambda item: (item["object"], item["class"]), seed=seed)
    rng = random.Random(seed)
    for name in objects:
        pool = sorted((item for item in normal if item["object"] == name), key=lambda item: item["image"])
        rng.shuffle(pool)
        for split_name, items in splits.items():
            count = round(sum(1 for item in items if item["object"] == name) * normal_ratio)
            splits[split_name] += [{**item, "boxes": []} for item in pool[:count]]
            pool = pool[count:]
    summary = {"objects": list(objects), "classes": visa.CLASSES, "skipped_empty_masks": len(skipped), "splits": {}}
    for split_name, items in splits.items():
        summary["splits"][split_name] = write_split(os.path.join(out, split_name), items, max_side)
    with open(os.path.join(out, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def write_split(folder, items, max_side):
    """Resized JPEGs and ``_annotations.coco.json`` of one split; returns its counts."""
    os.makedirs(folder, exist_ok=True)
    coco = {
        "images": [],
        "annotations": [],
        # 0-based ids: RF-DETR trains on the category ids as they are, and reads its predictions
        # as indexes into the class names
        "categories": [{"id": i - 1, "name": n, "supercategory": "none"} for i, n in visa.CLASSES.items()],
    }
    counts = Counter()
    for image_id, item in enumerate(sorted(items, key=lambda item: item["image"]), start=1):
        image = Image.open(item["image"]).convert("RGB")
        scale = min(1.0, max_side / max(image.size))
        if scale < 1.0:
            image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
        file_name = f"{item['object']}_{os.path.splitext(os.path.basename(item['image']))[0]}.jpg"
        image.save(os.path.join(folder, file_name), quality=92)
        coco["images"].append({"id": image_id, "file_name": file_name, "width": image.width, "height": image.height})
        counts["defective" if item["boxes"] else "good"] += 1
        for class_id, (x0, y0, x1, y1) in item["boxes"]:
            box = [x0 * scale, y0 * scale, (x1 - x0) * scale, (y1 - y0) * scale]
            coco["annotations"].append({
                "id": len(coco["annotations"]) + 1,
                "image_id": image_id,
                "category_id": class_id - 1,
                "bbox": [round(v, 2) for v in box],
                "area": round(box[2] * box[3], 2),
                "iscrowd": 0,
            })
            counts[visa.CLASSES[class_id]] += 1
    with open(os.path.join(folder, "_annotations.coco.json"), "w") as handle:
        json.dump(coco, handle)
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--visa", default="data/visa", help="extracted VisA (downloaded and extracted when missing)")
    parser.add_argument("--archive", default="data/VisA_20220922.tar")
    parser.add_argument("--out", default="dataset")
    parser.add_argument("--normal-ratio", type=float, default=1.0, help="good boards per defective board, per split")
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not all(os.path.exists(os.path.join(args.visa, name, "image_anno.csv")) for name in visa.OBJECTS):
        visa.extract(visa.download(args.archive), args.visa)
    summary = build(args.visa, args.out, normal_ratio=args.normal_ratio, max_side=args.max_side, seed=args.seed)
    print(json.dumps(summary["splits"], indent=2))
    print(f"COCO dataset in {args.out}/ ({summary['skipped_empty_masks']} masks without defects skipped)")


if __name__ == "__main__":
    main()
