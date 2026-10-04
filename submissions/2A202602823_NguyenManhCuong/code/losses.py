"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Mã hoàn thiện cho các hàm mất mát và phép trộn mẫu.
Liên hệ slide Day 2: label smoothing (trang 56), focal loss (trang 57), Mixup/CutMix (trang 48).

Giao diện bạn phải giữ:
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_criterion(kind: str = "ce", **kw):
    """Trả về hàm loss theo `kind`: "ce", "ls" (label smoothing), "focal", "ce_weighted".

    Ví dụ kw: smoothing=0.1, gamma=2.0, alpha=None, weight=tensor.
    TODO: tạo đúng loss, hoặc gọi các lớp bên dưới.
    """
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(kw.get("smoothing", 0.1))
    if kind == "focal":
        return FocalLoss(kw.get("gamma", 2.0), kw.get("alpha"))
    if kind == "ce_weighted":
        weight = kw.get("weight")
        if weight is None:
            raise ValueError("ce_weighted cần tensor weight")
        return nn.CrossEntropyLoss(weight=weight)
    raise ValueError(f"Loss không hợp lệ: {kind!r}")


class LabelSmoothingCE(nn.Module):  # TODO: kế thừa torch.nn.Module
    """Cross-entropy với label smoothing: q'(k) = (1 - eps) * 1[k == y] + eps / K  (slide trang 56).

    TODO: tự cài đặt hoặc dùng torch.nn.CrossEntropyLoss(label_smoothing=eps), rồi ghi rõ
    bạn đã chọn cách nào. Kiểm tra: eps = 0 phải cho đúng CE.
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing phải nằm trong [0, 1)")
        self.smoothing = smoothing

    def forward(self, logits, target):
        return F.cross_entropy(logits, target, label_smoothing=self.smoothing)


class FocalLoss(nn.Module):  # TODO: kế thừa torch.nn.Module
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)  (slide trang 57).

    TODO:
      - tính log_softmax, lấy p_t của lớp đúng, nhân (1 - p_t)^gamma, lấy trung bình batch
      - alpha: None hoặc vector trọng số theo lớp
    BẮT BUỘC viết một kiểm tra nhỏ: gamma = 0 phải cho đúng cross-entropy (sai số < 1e-6).
    """

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma không được âm")
        self.gamma = gamma
        if alpha is None:
            self.register_buffer("alpha", None)
        else:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits, target):
        log_prob = F.log_softmax(logits, dim=1)
        log_pt = log_prob.gather(1, target[:, None]).squeeze(1)
        pt = log_pt.exp()
        loss = -((1.0 - pt) ** self.gamma) * log_pt
        if self.alpha is not None:
            loss = loss * self.alpha[target]
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN.

    - beta = 0: trọng số tỉ lệ nghịch với số ảnh (1 / n_c), chuẩn hoá về trung bình 1
    - beta > 0: class-balanced theo "số mẫu hiệu dụng": w_c = (1 - beta) / (1 - beta ** n_c)
      (slide trang 57, Cui et al. arXiv:1901.05555); chuẩn hoá tổng trọng số về số lớp

    TODO: trả về tensor độ dài 9. Chỉ dùng số liệu của train, không dùng val hay test.
    """
    counts = torch.as_tensor(counts, dtype=torch.float64)
    if counts.ndim != 1 or (counts <= 0).any():
        raise ValueError("counts phải là vector và mọi lớp phải có ít nhất một ảnh")
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta phải nằm trong [0, 1)")
    if beta == 0.0:
        weights = 1.0 / counts
    else:
        weights = (1.0 - beta) / (1.0 - torch.pow(beta, counts))
    return (weights / weights.mean()).float()


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn một batch ảnh và nhãn.

    - lam ~ Beta(alpha, alpha)
    - mode="mixup": x_mix = lam * x + (1 - lam) * x[perm]
    - mode="cutmix": cắt một hộp chữ nhật từ x[perm] dán vào x, rồi điều chỉnh lam theo
      DIỆN TÍCH THỰC của hộp sau khi cắt ra ngoài biên (slide trang 48)
    - trả về (x_mix, (y_a, y_b, lam)) với y_a = y, y_b = y[perm]

    TODO: tự cài đặt. Kiểm tra bằng mắt: vẽ vài ảnh sau khi trộn và in lam.
    """
    if alpha <= 0:
        raise ValueError("alpha phải là số dương")
    if mode not in {"mixup", "cutmix"}:
        raise ValueError("mode phải là 'mixup' hoặc 'cutmix'")
    lam = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(x.size(0), device=x.device)

    if mode == "mixup":
        mixed = lam * x + (1.0 - lam) * x[permutation]
    else:
        _, _, height, width = x.shape
        cut_ratio = math.sqrt(1.0 - lam)
        cut_width, cut_height = int(width * cut_ratio), int(height * cut_ratio)
        center_x = int(torch.randint(width, (1,), device=x.device).item())
        center_y = int(torch.randint(height, (1,), device=x.device).item())
        x1 = max(center_x - cut_width // 2, 0)
        x2 = min(center_x + cut_width // 2, width)
        y1 = max(center_y - cut_height // 2, 0)
        y2 = min(center_y + cut_height // 2, height)
        mixed = x.clone()
        mixed[:, :, y1:y2, x1:x2] = x[permutation, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (width * height))
    return mixed, (y, y[permutation], lam)


def mixed_loss(criterion, logits, targets):
    """Loss cho batch đã trộn: lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b).

    TODO. Lưu ý: accuracy trên batch đã trộn không còn nghĩa bình thường; đánh giá bằng val.
    """
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
