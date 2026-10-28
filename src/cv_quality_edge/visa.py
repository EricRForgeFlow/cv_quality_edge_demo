"""The VisA dataset (Amazon, CC BY 4.0): download it and extract the objects we use."""

import os
import tarfile
import urllib.request

URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
OBJECTS = ("pcb1", "pcb2", "pcb3")
# Classes of the masks of PCB1-3 (utils/id2class.py of amazon-science/spot-diff), lowercase as
# the labels the model learns and sends to Odoo
CLASSES = {1: "bent", 2: "melt", 3: "scratch", 4: "missing"}


def download(dest, url=URL):
    """Download the archive to ``dest``, resuming a partial download; returns its path."""
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request) as response:
        size = int(response.headers["Content-Length"])
    done = os.path.getsize(dest) if os.path.exists(dest) else 0
    if done == size:
        return dest
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    request = urllib.request.Request(url, headers={"Range": f"bytes={done}-"})
    with urllib.request.urlopen(request) as response, open(dest, "ab") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            print(f"\rDownloading VisA: {done / size:6.1%}", end="", flush=True)
    print()
    if os.path.getsize(dest) != size:
        raise OSError(f"Incomplete download of {url}: {os.path.getsize(dest)} of {size} bytes")
    return dest


def extract(archive, dest, objects=OBJECTS):
    """Extract the folders of ``objects`` from the archive into ``dest``; returns ``dest``.

    The ``data`` filter refuses absolute paths, links out of ``dest`` and special files.
    """
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(archive) as tar:
        members = [m for m in tar if m.name.split("/")[0] in objects or _in_objects(m.name, objects)]
        tar.extractall(dest, members=members, filter="data")
    return dest


def _in_objects(name, objects):
    parts = name.split("/")
    return len(parts) > 1 and parts[1] in objects
