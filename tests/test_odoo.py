import base64
import datetime
import io
import json
import urllib.error
from unittest import mock

from PIL import Image

from cv_quality_edge import odoo
from cv_quality_edge.conveyor import tally
from cv_quality_edge.evaluate import board_metrics

DETECTIONS = [{"label": "scratch", "confidence": 0.9, "bbox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.1}}]


def test_payload_is_the_generic_format():
    body = odoo.payload(
        Image.new("RGB", (2560, 1600), "blue"),
        DETECTIONS,
        external_id="run-1-board.jpg",
        station="line1-cam-a",
        product_code="PCB-001",
        lot="PCB1",
        model_version="pcb_nano",
        captured_at=datetime.datetime(2026, 10, 28, 10, 0, tzinfo=datetime.UTC),
    )
    assert body["captured_at"] == "2026-10-28T10:00:00+00:00"
    assert body["model"] == {"name": "rf-detr-nano", "version": "pcb_nano"}
    assert body["detections"] == DETECTIONS
    assert (body["station"], body["product_code"], body["lot"]) == ("line1-cam-a", "PCB-001", "PCB1")
    image = Image.open(io.BytesIO(base64.b64decode(body["image"]["data"])))
    assert image.format == "JPEG" and max(image.size) == odoo.MAX_IMAGE_SIDE


def test_payload_leaves_out_empty_keys():
    body = odoo.payload(Image.new("RGB", (10, 10)), [], external_id="x")
    assert "station" not in body and "lot" not in body


class _Response(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_send_headers_and_answer():
    client = odoo.Odoo("http://odoo:8069/", "line1", "secret", db="vision_demo")
    answer = _Response(json.dumps({"inspection": "VIS/1", "verdict": "fail"}).encode())
    with mock.patch("urllib.request.urlopen", return_value=answer) as urlopen:
        assert client.send({"detections": []}) == (200, {"inspection": "VIS/1", "verdict": "fail"})
    request = urlopen.call_args.args[0]
    assert request.full_url == "http://odoo:8069/vision_inspection/line1/results"
    assert request.get_header("Authorization") == "Bearer secret"
    assert request.get_header("X-odoo-database") == "vision_demo"
    assert request.get_header("Content-type") == "application/json"


def test_send_returns_odoo_errors():
    client = odoo.Odoo("http://odoo:8069", "line1", "wrong")
    error = urllib.error.HTTPError("url", 401, "Unauthorized", {}, io.BytesIO(b'{"error": "Invalid token"}'))
    with mock.patch("urllib.request.urlopen", side_effect=error):
        assert client.send({}) == (401, {"error": "Invalid token"})


def test_tally():
    results = [(True, "fail"), (True, "review"), (True, "pass"), (False, "pass"), (False, "review"), (False, None)]
    assert tally(results) == {
        "defective_failed": 1, "defective_review": 1, "defective_passed": 1,
        "good_passed": 1, "good_review": 1, "not_sent": 1,
    }


def test_board_metrics():
    metrics = board_metrics([True, True, False, False], [True, False, True, False])
    assert (metrics["precision"], metrics["recall"], metrics["accuracy"]) == (0.5, 0.5, 0.5)
