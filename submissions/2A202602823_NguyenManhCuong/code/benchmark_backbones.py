"""Đo batch-1 latency cho năm backbone đã so sánh và lưu CSV."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import torch


CODE_DIR = Path(__file__).resolve().parent
STARTER_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

import benchmark
import model


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for exp_id in ("B01", "B02", "B03", "B04", "B05"):
        run_dir = STARTER_DIR / "runs" / exp_id / "seed0"
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        network = model.build_model(
            cfg["backbone"], pretrained=False, num_classes=9,
            drop_rate=cfg.get("drop_rate", 0.0), init="scratch",
        )
        report = benchmark.latency_report(
            network, batch_size=1, img_size=cfg["img_size"], dtype="fp32",
            device=device, warmup=10, iters=100,
        )
        rows.append({"exp_id": exp_id, "backbone": cfg["backbone"], **report})
        print(f"{exp_id}: p50={report['p50']:.3f} ms, p95={report['p95']:.3f} ms")
        del network
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(STARTER_DIR / "backbone_latency.csv", index=False)


if __name__ == "__main__":
    main()
