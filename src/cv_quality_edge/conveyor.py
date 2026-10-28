"""A simulated conveyor: boards pass under the camera, the model inspects them, Odoo decides.

    cv-conveyor --limit 20                     # to the Odoo of the environment variables
    cv-conveyor --dry-run --interval 0         # detection only, nothing sent

The boards are the test split of the dataset, never seen in training, in a random order.
Each one is inspected, saved with its boxes under runs/conveyor-<time>/ and sent to Odoo;
at the end, Odoo's verdicts are compared with what each board really was.

Odoo settings come from the options or the environment: CV_ODOO_URL, CV_ODOO_DB,
CV_ODOO_SOURCE, CV_ODOO_TOKEN, CV_ODOO_STATION.
"""

import argparse
import datetime
import os
import random
import time
from collections import Counter

from PIL import Image

from . import odoo as odoo_client
from .detector import DEFAULT_MODEL, Detector, annotate, describe
from .evaluate import load_split


def boards(dataset, split="test", seed=None, limit=None):
    """``[(path, defective)]`` of the boards of a split, shuffled."""
    items = [(path, bool(boxes)) for path, boxes in load_split(dataset, split)]
    random.Random(seed).shuffle(items)
    return items[:limit] if limit else items


def tally(results):
    """Odoo's verdicts against the truth: ``[(defective, verdict)]`` → a summary dict.

    A defective board is caught when Odoo doesn't pass it (fail, or review by a person).
    """
    counts = Counter()
    for defective, verdict in results:
        if verdict is None:
            counts["not_sent"] += 1
        elif defective:
            counts["defective_failed" if verdict == "fail" else
                   "defective_review" if verdict == "review" else "defective_passed"] += 1
        else:
            counts["good_passed" if verdict == "pass" else
                   "good_review" if verdict == "review" else "good_failed"] += 1
    return dict(counts)


def run(args):
    detector = Detector(args.model, args.threshold)
    odoo = None if args.dry_run else odoo_client.Odoo(args.odoo_url, args.source, args.token, args.db)
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = os.path.join("runs", f"conveyor-{run_id}")
    os.makedirs(out, exist_ok=True)
    line = boards(args.dataset, seed=args.seed, limit=args.limit)
    results = []
    for number, (path, defective) in enumerate(line, start=1):
        started = time.monotonic()
        name = os.path.basename(path)
        image = Image.open(path)
        detect_start = time.perf_counter()
        detections = detector.detect(image)
        latency = (time.perf_counter() - detect_start) * 1000
        annotate(image, detections).save(os.path.join(out, f"{number:03d}_{name}"))
        verdict, answer = None, "not sent"
        if odoo:
            status, response = odoo.send(odoo_client.payload(
                image,
                detections,
                external_id=f"{run_id}-{name}",
                station=args.station,
                product_code=args.product_code,
                lot=name.split("_")[0].upper(),
                model_version=detector.version,
            ))
            verdict = response.get("verdict") if status == 200 else None
            answer = f"{response.get('inspection')} {verdict.upper()}" if verdict else f"error {status}: {response.get('error')}"
        truth = "defective" if defective else "good"
        print(f"[{number:3d}/{len(line)}] {name:22s} {truth:9s} | {describe(detections)} ({latency:.0f} ms) | Odoo: {answer}")
        results.append((defective, verdict))
        wait = args.interval - (time.monotonic() - started)
        if wait > 0 and number < len(line):
            time.sleep(wait)
    summary = tally(results)
    print(f"\nAnnotated boards in {out}/")
    if odoo:
        caught = summary.get("defective_failed", 0) + summary.get("defective_review", 0)
        defective = caught + summary.get("defective_passed", 0)
        good = sum(v for k, v in summary.items() if k.startswith("good_"))
        print(f"Defective boards stopped: {caught}/{defective} "
              f"({summary.get('defective_review', 0)} of them sent to review)")
        print(f"Good boards passed: {summary.get('good_passed', 0)}/{good} "
              f"({summary.get('good_review', 0)} sent to review, {summary.get('good_failed', 0)} rejected)")
        if summary.get("not_sent"):
            print(f"Boards Odoo refused: {summary['not_sent']}")
    return summary


def main():
    env = os.environ.get
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--interval", type=float, default=3.0, help="seconds between two boards")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--odoo-url", default=env("CV_ODOO_URL", "http://localhost:20069"))
    parser.add_argument("--db", default=env("CV_ODOO_DB"))
    parser.add_argument("--source", default=env("CV_ODOO_SOURCE", "line1"))
    parser.add_argument("--token", default=env("CV_ODOO_TOKEN"))
    parser.add_argument("--station", default=env("CV_ODOO_STATION"))
    parser.add_argument("--product-code", default="PCB-001")
    parser.add_argument("--dry-run", action="store_true", help="detect only, send nothing to Odoo")
    args = parser.parse_args()
    if not args.dry_run and not args.token:
        parser.error("give the source's token (--token or CV_ODOO_TOKEN), or use --dry-run")
    run(args)


if __name__ == "__main__":
    main()
