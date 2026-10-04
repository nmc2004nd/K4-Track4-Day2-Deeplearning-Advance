"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Mã hoàn thiện cho đo độ trễ và thông lượng.

Quy tắc đo (vi phạm bị trừ điểm, RUBRIC mục 3):
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() (hoặc CUDA event) TRƯỚC và SAU đoạn cần đo
  - >= 50 lần đo, báo cáo p50, p95, p99 (không chỉ trung bình)
  - ghi rõ GPU, dtype (FP32/AMP/FP16), batch, độ phân giải, có/không gộp BN, phiên bản torch
  - chọn và ghi rõ có tính tiền xử lý hay không
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian một hàm `fn()` (không tham số), trả về mili-giây.

    `sync` là hàm đồng bộ (ví dụ torch.cuda.synchronize) hoặc None trên CPU.

    TODO:
      - chạy warmup lần đầu rồi bỏ
      - với mỗi lần đo: sync(); t0 = time.perf_counter(); fn(); sync(); lấy hiệu * 1000
      - trả về {"p50": ..., "p95": ..., "p99": ..., "mean": ..., "n": iters}
    Gợi ý: dùng numpy.percentile hoặc torch.quantile.
    """
    if warmup < 10:
        raise ValueError("warmup phải >= 10")
    if iters < 50:
        raise ValueError("iters phải >= 50")
    for _ in range(warmup):
        fn()
    if sync is not None:
        sync()

    times_ms = []
    for _ in range(iters):
        if sync is not None:
            sync()
        start = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        times_ms.append((time.perf_counter() - start) * 1000.0)
    values = np.asarray(times_ms)
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "mean": float(values.mean()),
        "n": int(iters),
    }


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size).

    Trả về dict có thể ghi thẳng vào sheet `Latency` của results.xlsx:
        {"gpu": ..., "dtype": ..., "batch": ..., "img_size": ..., "p50": ..., "p95": ..., "p99": ...,
         "images_per_s": batch_size / (p50 / 1000), "torch": torch.__version__}

    TODO:
      - model.eval(), torch.inference_mode()
      - dtype: "fp32" | "amp" (autocast) | "fp16" (model.half())
      - gọi bench(...) với sync phù hợp; lấy tên GPU bằng torch.cuda.get_device_name
      - Nhớ: ở batch 1, AMP có thể CHẬM hơn FP32 (slide trang 73): đo thật, đừng giả định
    """
    if dtype not in {"fp32", "amp", "fp16"}:
        raise ValueError("dtype phải là fp32, amp hoặc fp16")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA không khả dụng")
    if dtype == "fp16" and target_device.type != "cuda":
        raise ValueError("fp16 benchmark chỉ hỗ trợ CUDA")

    benchmark_model = copy.deepcopy(model).to(target_device).eval()
    input_dtype = torch.float16 if dtype == "fp16" else torch.float32
    if dtype == "fp16":
        benchmark_model.half()
    sample = torch.randn(batch_size, 3, img_size, img_size, device=target_device, dtype=input_dtype)
    amp_enabled = dtype == "amp" and target_device.type == "cuda"

    def forward():
        with torch.inference_mode(), torch.autocast(
            device_type=target_device.type, enabled=amp_enabled
        ):
            benchmark_model(sample)

    sync = torch.cuda.synchronize if target_device.type == "cuda" else None
    stats = bench(forward, warmup=warmup, iters=iters, sync=sync)
    stats.update({
        "gpu": torch.cuda.get_device_name(target_device) if target_device.type == "cuda" else "CPU",
        "dtype": dtype,
        "batch": batch_size,
        "img_size": img_size,
        "images_per_s": batch_size / (stats["p50"] / 1000.0),
        "torch": torch.__version__,
        "includes_preprocessing": False,
    })
    del benchmark_model, sample
    return stats


def tta_latency(model, k_views: int, **kw) -> dict:
    """Độ trễ của TTA K view: xấp xỉ K lần một lượt chạy (slide trang 63). TODO: đo thật, so với K * p50."""
    if k_views < 1:
        raise ValueError("k_views phải >= 1")

    class RepeatViews(nn.Module):
        def __init__(self, inner, repeats):
            super().__init__()
            self.inner = inner
            self.repeats = repeats

        def forward(self, x):
            outputs = [self.inner(x if i == 0 else torch.flip(x, dims=(-1,)))
                       for i in range(self.repeats)]
            return torch.stack(outputs).mean(0)

    report = latency_report(RepeatViews(model, k_views), **kw)
    report["k_views"] = k_views
    return report
