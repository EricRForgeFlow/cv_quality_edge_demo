"""Client of Odoo's Vision Inspection endpoint (module vision_inspection, generic format)."""

import base64
import datetime
import io
import json
import urllib.error
import urllib.request

MAX_IMAGE_SIDE = 1280


class Odoo:
    """A vision source of Odoo: ``POST <url>/vision_inspection/<source>/results``."""

    def __init__(self, url, source, token, db=None, timeout=30):
        self.endpoint = f"{url.rstrip('/')}/vision_inspection/{source}/results"
        self.token = token
        self.db = db
        self.timeout = timeout

    def send(self, payload):
        """POST the results; returns ``(status, response)``, the response as a dict.

        Odoo's errors (401 wrong token, 404 unknown station, 400 invalid results...) are
        returned, not raised: a line keeps running when one board is refused.
        """
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.token}"}
        if self.db:
            headers["X-Odoo-Database"] = self.db
        request = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode(), headers=headers,
                                         method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            try:
                return error.code, json.load(error)
            except ValueError:
                return error.code, {"error": error.reason}


def image_data(image, max_side=MAX_IMAGE_SIDE):
    """The image as base64 JPEG, at most ``max_side`` pixels wide or high."""
    image = image.convert("RGB")
    scale = min(1.0, max_side / max(image.size))
    if scale < 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode()


def payload(image, detections, external_id, station=None, product_code=None, lot=None, model_name="rf-detr-nano",
            model_version=None, captured_at=None):
    """The results of one board in vision_inspection's generic format."""
    captured_at = captured_at or datetime.datetime.now(datetime.UTC)
    body = {
        "external_id": external_id,
        "captured_at": captured_at.isoformat(timespec="seconds"),
        "model": {"name": model_name, "version": model_version},
        "image": {"mime": "image/jpeg", "data": image_data(image)},
        "detections": detections,
    }
    for key, value in (("station", station), ("product_code", product_code), ("lot", lot)):
        if value:
            body[key] = value
    return body
