"""Fine-tune RF-DETR on the circuit boards (dataset/ from cv-prepare).

    cv-train --preset colab          # on a GPU (what the Colab notebook runs)
    cv-train --preset cpu            # the fallback without a GPU: slower, smaller images

The best weights are copied to models/pcb_<model>.pth, where cv-detect and cv-conveyor
look for them.
"""

import argparse
import os
import shutil

PRESETS = {
    "colab": {"resolution": 640, "epochs": 60, "batch_size": "auto", "grad_accum_steps": 1, "num_workers": 2},
    "cpu": {"resolution": 448, "epochs": 30, "batch_size": 2, "grad_accum_steps": 8, "num_workers": 0, "device": "cpu"},
}
# The best checkpoint RF-DETR keeps, in order of preference
BEST = ("checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best_regular.pth")


def model_class(name):
    from rfdetr import RFDETRNano, RFDETRSmall

    return {"nano": RFDETRNano, "small": RFDETRSmall}[name]


def train(dataset="dataset", output="output", model="nano", preset="colab", lr=1e-4, early_stopping=True,
          models_dir="models", **overrides):
    """Train and return the path of the kept weights."""
    settings = {**PRESETS[preset], **{k: v for k, v in overrides.items() if v is not None}}
    model_kwargs = {"resolution": settings.pop("resolution")}
    if "device" in settings:
        model_kwargs["device"] = settings.pop("device")
    detector = model_class(model)(**model_kwargs)
    detector.train(
        dataset_dir=dataset,
        output_dir=output,
        lr=lr,
        early_stopping=early_stopping,
        **settings,
    )
    best = next((os.path.join(output, name) for name in BEST if os.path.exists(os.path.join(output, name))), None)
    if not best:
        raise FileNotFoundError(f"No best checkpoint in {output}/ (looked for {', '.join(BEST)})")
    os.makedirs(models_dir, exist_ok=True)
    kept = os.path.join(models_dir, f"pcb_{model}.pth")
    shutil.copy2(best, kept)
    print(f"{best} -> {kept}")
    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--output", default="output")
    parser.add_argument("--model", choices=("nano", "small"), default="nano")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="colab")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resolution", type=int, help="square input size, a multiple of 32")
    parser.add_argument("--batch-size", type=lambda v: v if v == "auto" else int(v))
    parser.add_argument("--grad-accum-steps", type=int)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no-early-stopping", action="store_true")
    args = parser.parse_args()
    train(
        dataset=args.dataset,
        output=args.output,
        model=args.model,
        preset=args.preset,
        lr=args.lr,
        early_stopping=not args.no_early_stopping,
        epochs=args.epochs,
        resolution=args.resolution,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
    )


if __name__ == "__main__":
    main()
