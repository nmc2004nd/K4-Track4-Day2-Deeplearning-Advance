"""Sinh prediction validation của cấu hình khóa F01 = T00 + I04 cho ba seed."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F


CODE_DIR = Path(__file__).resolve().parent
STARTER_DIR = CODE_DIR.parent
REPO_DIR = STARTER_DIR.parent
for path in (CODE_DIR, REPO_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import dataset
import inference
import model
from eval import check_against_csv, read_pred, save_predictions


def load_model(seed: int, device: torch.device):
    run_dir = STARTER_DIR / "runs" / "T00" / f"seed{seed}"
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    network = model.build_model(
        cfg["backbone"], pretrained=False, num_classes=dataset.NUM_CLASSES,
        drop_rate=cfg.get("drop_rate", 0.0), init="scratch",
    ).to(device)
    try:
        state = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(run_dir / "best.pt", map_location=device)
    network.load_state_dict(state)
    return network.eval(), cfg


def main() -> None:
    selected = json.loads((STARTER_DIR / "best_inference.json").read_text(encoding="utf-8"))
    if selected["training_exp"] != "T00" or selected["inference_exp"] != "I04":
        raise RuntimeError(f"Cấu hình khóa không phải T00 + I04: {selected}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = STARTER_DIR / "predictions"
    output_dir.mkdir(exist_ok=True)

    for seed in (0, 1, 2):
        output_path = output_dir / f"F01_seed{seed}_val.csv"
        network, cfg = load_model(seed, device)
        _, val_df, _ = dataset.load_split(cfg["labels_dir"], cfg["fold"])
        val_csv = Path(cfg["labels_dir"]) / f"val_subset{cfg['fold']}.csv"
        transform = dataset.build_transforms(False, cfg["img_size"], "basic")
        loader = dataset.make_loader(
            val_df, cfg["images_dir"], transform,
            batch_size=64, train=False, num_workers=0,
        )
        filenames, labels, logits = inference.predict_logits(
            network, loader, device,
            view=lambda images: F.interpolate(
                images, size=(256, 256), mode="bilinear", align_corners=False
            ),
        )
        probabilities = inference.aggregate_views([logits], space="prob")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
            raise AssertionError("Xác suất F01 val không được chuẩn hóa")
        save_predictions(output_path, filenames, labels, probabilities)
        check_against_csv(read_pred(str(output_path)), str(val_csv), what="val")
        print(f"Saved {output_path.name}: {len(filenames)} rows")
        del network
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
