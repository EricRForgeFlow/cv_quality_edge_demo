import argparse
import json
from unittest import mock

from PIL import Image

from cv_quality_edge import conveyor

SCRATCH = {"label": "scratch", "confidence": 0.9, "bbox": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}}


class FakeDetector:
    version = "fake"

    def __init__(self, *args, **kwargs):
        pass

    def detect(self, image):
        red, green, _blue = image.convert("RGB").getpixel((20, 15))
        return [SCRATCH] if red > 200 and green < 80 else []


def make_split(folder):
    """Two boards: a red one with a box (defective), a green one without (good)."""
    (folder / "valid").mkdir(parents=True)
    Image.new("RGB", (40, 30), "red").save(folder / "valid" / "pcb1_001.jpg")
    Image.new("RGB", (40, 30), "green").save(folder / "valid" / "pcb1_0002.jpg")
    (folder / "valid" / "_annotations.coco.json").write_text(json.dumps({
        "images": [{"id": 1, "file_name": "pcb1_001.jpg", "width": 40, "height": 30},
                   {"id": 2, "file_name": "pcb1_0002.jpg", "width": 40, "height": 30}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 2, "bbox": [4, 3, 8, 6], "area": 48, "iscrowd": 0}],
        "categories": [{"id": 2, "name": "scratch", "supercategory": "none"}],
    }))


def test_jsonl_events(tmp_path, monkeypatch, capsys):
    make_split(tmp_path / "dataset")
    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(
        model="m.pth", threshold=0.4, dataset="dataset", split="valid", seed=1, limit=None, interval=0,
        dry_run=False, jsonl=True, odoo_url="http://odoo", source="line1", token="t", db="db",
        station="cam", product_code="PCB-001",
    )
    answers = {"pcb1_001.jpg": (200, {"inspection": "VIS/1", "verdict": "fail"}),
               "pcb1_0002.jpg": (200, {"inspection": "VIS/2", "verdict": "pass"})}
    with mock.patch.object(conveyor, "Detector", FakeDetector), \
         mock.patch("cv_quality_edge.odoo.Odoo.send", side_effect=lambda body: answers[body["external_id"].split("-", 2)[-1]]):
        summary = conveyor.run(args)
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    boards = {event["board"]: event for event in lines[:-1]}
    assert boards["pcb1_001.jpg"]["defective"] and boards["pcb1_001.jpg"]["verdict"] == "fail"
    assert boards["pcb1_001.jpg"]["detections"] == [SCRATCH]
    assert boards["pcb1_0002.jpg"]["inspection"] == "VIS/2"
    assert (tmp_path / boards["pcb1_001.jpg"]["annotated"]).exists()
    assert lines[-1] == {"summary": summary} == {"summary": {"defective_failed": 1, "good_passed": 1}}
