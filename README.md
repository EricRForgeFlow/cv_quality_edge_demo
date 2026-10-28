# cv_quality_edge_demo

A circuit board defect detector, trained on a public dataset, running on a CPU next to a
(simulated) conveyor and sending what it finds to **Odoo's Vision Inspection** (modules
`vision_inspection` and `vision_inspection_quality_control_oca`, repository
`odoo_custom_addons`). No camera needed: the boards are photos of the
[VisA](https://github.com/amazon-science/spot-diff) dataset.

```
VisA photos ──▶ cv-conveyor ──▶ RF-DETR Nano (CPU) ──▶ POST /vision_inspection/<source>/results ──▶ Odoo
                 one board         bent, melt,              image + boxes + confidences             verdict, OCA
                 every few s       scratch, missing                                                  quality inspection
```

| | |
| --- | --- |
| Model | [RF-DETR](https://github.com/roboflow/rf-detr) **Nano** (Apache-2.0), fine-tuned from its COCO weights |
| Data | VisA PCB1, PCB2, PCB3 (CC BY 4.0): 300 defective boards (masks turned into boxes) + good boards |
| Classes | `bent`, `melt`, `scratch`, `missing` (the dataset's own) |
| Training | Google Colab (GPU), or a CPU as a fallback |
| Licence | Apache-2.0. Every dependency allows commercial use; see [NOTICE](NOTICE) |

## Quick start

Needs [uv](https://docs.astral.sh/uv/). Locally, torch comes from its CPU build.

```bash
uv sync                      # detecting and the conveyor
uv sync --extra train        # + training (PyTorch Lightning, COCO tools)
uv run pytest
```

### 1. Train on Colab (recommended)

Open [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) in Colab
(*File → Open notebook → GitHub*, this repository), select a **T4 GPU** and *Run all*. It
downloads VisA, builds the dataset, trains, scores the model on the test boards and saves
`pcb_nano.pth` (+ its scores) to Google Drive. Put the `.pth` in `models/` here.

### 1b. Or train on a CPU (fallback)

```bash
uv run cv-prepare            # downloads VisA (1.9 GB) to data/, builds dataset/
uv run cv-train --preset cpu # smaller images (448 px), batch 2 × 8 accumulated
uv run cv-evaluate
```

One epoch takes about 20 minutes on a 12-core laptop CPU, so the 30 epochs of the preset
run overnight. On a CPU, detection then takes 100 to 300 ms per board.

### 2. Run the conveyor

```bash
uv run cv-prepare                                    # if dataset/ isn't there yet
uv run cv-detect dataset/test/pcb1_000.jpg           # one board: runs/detect/
uv run cv-conveyor --dry-run --limit 10 --interval 0 # detection only

export CV_ODOO_URL=http://localhost:20069 CV_ODOO_DB=vision_demo \
       CV_ODOO_SOURCE=line1 CV_ODOO_TOKEN=line1-demo-token CV_ODOO_STATION=line1-cam-a
uv run cv-conveyor --limit 20                        # one board every 3 s, sent to Odoo
```

Each board is printed with what the model found, its latency and Odoo's verdict, and
saved with its boxes under `runs/conveyor-<time>/`. At the end, Odoo's verdicts are compared
with what each board really was.

In Odoo, the source needs a station (`CV_ODOO_STATION`) and the product `PCB-001`. The
`melt` class is created by Odoo the first time the model sends it, allowing none per
board: rename it *Melt* and set its severity under *Vision Inspection → Configuration →
Defect Classes*.

## How the dataset is made

`cv-prepare` (`src/cv_quality_edge/prepare.py`):

- VisA's masks store a class per pixel (1 bent, 2 melt, 3 scratch, 4 missing). Each
  connected region of a class becomes a box; regions a few pixels apart are one defect,
  specks under 30 px are dropped.
- Good boards are added as images without boxes, as many as defective ones in each
  split, so the model also learns to find nothing.
- Split 70 / 15 / 15 (train / valid / test), stratified by board type and main defect,
  with a fixed seed. Images are resized to 1024 px.

| Split | Defective | Good | bent | melt | scratch | missing |
| --- | --- | --- | --- | --- | --- | --- |
| train | 210 | 210 | 50 | 214 | 67 | 77 |
| valid | 45 | 45 | 11 | 47 | 15 | 14 |
| test | 45 | 45 | 9 | 45 | 18 | 25 |

(the class columns count boxes: a board can have several defects)

## Results

To be filled in after the Colab run (`models/pcb_nano.metrics.json`).

## Layout

| Path | |
| --- | --- |
| `src/cv_quality_edge/visa.py` | download VisA, extract PCB1-3 |
| `src/cv_quality_edge/prepare.py` | `cv-prepare`: masks → boxes, splits, COCO files |
| `src/cv_quality_edge/train.py` | `cv-train`: fine-tuning, presets `colab` and `cpu` |
| `src/cv_quality_edge/detector.py` | `cv-detect`: load the weights, detect, draw the boxes |
| `src/cv_quality_edge/evaluate.py` | `cv-evaluate`: mAP and pass/fail per board on the test split |
| `src/cv_quality_edge/odoo.py` | client of Odoo's Vision Inspection endpoint |
| `src/cv_quality_edge/conveyor.py` | `cv-conveyor`: the simulated line |
| `notebooks/train_colab.ipynb` | training on Colab |

`data/`, `dataset/`, `models/`, `output/` and `runs/` are downloaded or generated, and not
committed.
