import json

import numpy as np
from PIL import Image

from cv_quality_edge import prepare


def test_one_box_per_region_and_class():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:20, 10:30] = 3
    mask[60:80, 50:60] = 3
    mask[40:45, 80:90] = 1
    boxes = sorted(prepare.mask_boxes(mask, min_area=1, gap=0))
    assert boxes == [(1, (80, 40, 90, 45)), (3, (10, 10, 30, 20)), (3, (50, 60, 60, 80))]


def test_close_regions_merge_and_specks_drop():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:20, 10:20] = 2
    mask[10:20, 23:30] = 2  # 3 px apart: the same defect
    mask[90:92, 90:92] = 2  # 4 px: noise
    assert prepare.mask_boxes(mask, min_area=10, gap=3) == [(2, (10, 10, 30, 20))]


def test_box_extent_ignores_the_merge_margin():
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[20:30, 20:30] = 4
    assert prepare.mask_boxes(mask, min_area=1, gap=5) == [(4, (20, 20, 30, 30))]


def test_split_is_stratified_and_reproducible():
    items = [{"image": f"{o}/{i:03d}", "object": o, "class": c}
             for o in ("pcb1", "pcb2") for c in (1, 2) for i in range(20)]
    result = prepare.split(items, key=lambda item: (item["object"], item["class"]), seed=7)
    assert sum(len(v) for v in result.values()) == len(items)
    for name in ("valid", "test"):
        groups = {(item["object"], item["class"]) for item in result[name]}
        assert len(groups) == 4, f"every group is in {name}"
    again = prepare.split(items, key=lambda item: (item["object"], item["class"]), seed=7)
    assert [i["image"] for i in again["test"]] == [i["image"] for i in result["test"]]


def test_build_writes_coco(tmp_path):
    root = tmp_path / "visa"
    for name in ("pcb1",):
        for kind in ("Normal", "Anomaly"):
            (root / name / "Data" / "Images" / kind).mkdir(parents=True)
        (root / name / "Data" / "Masks" / "Anomaly").mkdir(parents=True)
        rows = ["image,label,mask"]
        for i in range(6):
            Image.new("RGB", (200, 100), "green").save(root / name / "Data" / "Images" / "Normal" / f"{i:04d}.JPG")
            rows.append(f"{name}/Data/Images/Normal/{i:04d}.JPG,normal,")
            mask = np.zeros((100, 200), dtype=np.uint8)
            mask[20:40, 50:90] = 3
            Image.new("RGB", (200, 100), "green").save(root / name / "Data" / "Images" / "Anomaly" / f"{i:03d}.JPG")
            Image.fromarray(mask).save(root / name / "Data" / "Masks" / "Anomaly" / f"{i:03d}.png")
            rows.append(f"{name}/Data/Images/Anomaly/{i:03d}.JPG,scratch,{name}/Data/Masks/Anomaly/{i:03d}.png")
        (root / name / "image_anno.csv").write_text("\n".join(rows) + "\n")
    summary = prepare.build(str(root), str(tmp_path / "out"), objects=("pcb1",), max_side=100, seed=1)
    assert summary["splits"]["train"]["defective"] == summary["splits"]["train"]["good"]
    coco = json.loads((tmp_path / "out" / "test" / "_annotations.coco.json").read_text())
    assert [(c["id"], c["name"]) for c in coco["categories"]] == [(0, "bent"), (1, "melt"), (2, "scratch"), (3, "missing")]
    annotation = coco["annotations"][0]
    assert annotation["category_id"] == 2, "scratch: mask value 3, 0-based category 2"
    assert annotation["bbox"] == [25.0, 10.0, 20.0, 10.0], "boxes follow the image to its new size"
    assert all(image["width"] == 100 for image in coco["images"])
