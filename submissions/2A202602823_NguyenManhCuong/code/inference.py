"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Mã hoàn thiện cho suy luận, TTA, ensemble và hiệu chuẩn.
Liên hệ slide Day 2: TTA (trang 62-66, 75), ensemble/EMA/soup (trang 67), độ phân giải kiểm tra
(trang 68), temperature scaling (trang 69), gộp BatchNorm (trang 71).

Mọi hàm phải chạy ở chế độ eval, không gradient. Chọn phương pháp CHỈ dựa trên val;
nhiệt độ T khớp trên VAL rồi áp dụng sang test (README.md, S2 và S4).

Giao diện bạn nên giữ:
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F


def predict_logits(model, loader, device, view=None):
    """Chạy model trên loader và gom logit theo đúng thứ tự file.

    `view` là hàm biến đổi batch ảnh trước khi đưa vào model (ví dụ lật ngang), hoặc None.
    TODO: model.eval(), torch.inference_mode(), (tuỳ chọn) autocast. Trả về numpy.
    """
    device = torch.device(device)
    model.eval()
    filenames, labels_all, logits_all = [], [], []
    transform_view = view or view_identity
    with torch.inference_mode():
        for images, labels, batch_filenames in loader:
            images = images.to(device, non_blocking=True)
            images = transform_view(images)
            logits = model(images)
            filenames.extend(list(batch_filenames))
            labels_all.append(labels.cpu())
            logits_all.append(logits.float().cpu())
    return filenames, torch.cat(labels_all).numpy(), torch.cat(logits_all).numpy()


def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W). TODO: dùng torch.flip trên chiều rộng (slide trang 75)."""
    return torch.flip(x, dims=(-1,))


def views_multicrop(x, crop: int):
    """5 crop (4 góc + giữa) kích thước `crop`, và tuỳ chọn thêm bản lật. Trả về list các batch. TODO."""
    if x.ndim != 4:
        raise ValueError("x phải có dạng (N, C, H, W)")
    height, width = x.shape[-2:]
    if crop <= 0 or crop > min(height, width):
        raise ValueError(f"crop={crop} không hợp lệ với ảnh {height}x{width}")
    top, left = 0, 0
    bottom, right = height - crop, width - crop
    center_y, center_x = (height - crop) // 2, (width - crop) // 2
    positions = [
        (top, left), (top, right), (bottom, left), (bottom, right),
        (center_y, center_x),
    ]
    return [x[:, :, y:y + crop, x0:x0 + crop] for y, x0 in positions]


def views_multiscale(x, sizes):
    """Resize batch về từng kích thước trong `sizes`, trả về list các batch. TODO.

    Lưu ý: model phải chấp nhận ảnh khác kích thước lúc train (CNN có global pooling thì được;
    ViT/Swin cần xử lý riêng vị trí/cửa sổ). Ghi rõ giới hạn bạn gặp.
    """
    if x.ndim != 4:
        raise ValueError("x phải có dạng (N, C, H, W)")
    outputs = []
    for size in sizes:
        if isinstance(size, int):
            target = (size, size)
        else:
            target = tuple(size)
        if min(target) <= 0:
            raise ValueError(f"Kích thước không hợp lệ: {size}")
        outputs.append(F.interpolate(x, size=target, mode="bilinear", align_corners=False))
    return outputs


def aggregate_views(logits_per_view, space: str = "prob"):
    """Gộp K lượt chạy của TTA thành một dự đoán (slide trang 62).

      - space="prob":  trung bình softmax của từng view
      - space="logit": trung bình logit rồi softmax
    Slide chưa kết luận cách nào luôn tốt hơn: chọn một và ghi rõ, hoặc so sánh cả hai (I03).
    TODO: trả về xác suất (N, 9) đã chuẩn hoá.
    """
    if not logits_per_view:
        raise ValueError("Cần ít nhất một tensor/array logits")
    tensors = [torch.as_tensor(item, dtype=torch.float32) for item in logits_per_view]
    if any(tensor.shape != tensors[0].shape for tensor in tensors[1:]):
        raise ValueError("Logits của các view phải cùng shape")
    stacked = torch.stack(tensors)
    if space == "prob":
        probabilities = stacked.softmax(dim=-1).mean(dim=0)
    elif space == "logit":
        probabilities = stacked.mean(dim=0).softmax(dim=-1)
    else:
        raise ValueError("space phải là 'prob' hoặc 'logit'")
    return probabilities.numpy()


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình (khác backbone hoặc khác seed). TODO.

    Chi phí suy luận = số mô hình. Chỉ ghép các mô hình trên CÙNG tập ảnh và cùng thứ tự file.
    """
    if not list_of_probs:
        raise ValueError("Cần ít nhất một ma trận xác suất")
    arrays = [np.asarray(probabilities, dtype=np.float64) for probabilities in list_of_probs]
    if any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError("Xác suất của các model phải cùng shape")
    result = np.mean(arrays, axis=0)
    return result / result.sum(axis=1, keepdims=True)


def fit_temperature(val_logits, val_labels) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu NLL trên VAL: p = softmax(logit / T)  (slide trang 69).

    TODO: tối ưu hoá một tham số (LBFGS trên log T, hoặc tìm lưới thô rồi tinh).
    Accuracy không đổi vì thứ tự lớp không đổi. KHÔNG khớp T trên test.
    """
    logits = torch.as_tensor(val_logits, dtype=torch.float64)
    labels = torch.as_tensor(val_labels, dtype=torch.long)
    if logits.ndim != 2 or len(logits) != len(labels):
        raise ValueError("val_logits và val_labels không cùng kích thước")
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.1, max_iter=100, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_temperature.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0))


def apply_temperature(logits, T: float):
    """Trả về softmax(logits / T). TODO."""
    if not np.isfinite(T) or T <= 0:
        raise ValueError("T phải là số hữu hạn và dương")
    tensor = torch.as_tensor(logits, dtype=torch.float32)
    return (tensor / T).softmax(dim=-1).numpy()


def fuse_conv_bn(model):
    """Gộp BatchNorm vào tích chập liền trước, chính xác lúc suy luận (slide trang 71, 75):

        w' = gamma * w / sqrt(var + eps)        b' = beta + gamma * (b - mean) / sqrt(var + eps)

    TODO:
      - model.eval() trước
      - với từng cặp (Conv2d, BatchNorm2d) liền kề: tạo conv mới (có bias) và thay BN bằng Identity
      - kiểm tra: đầu ra trước/sau gộp lệch nhau cỡ 1e-5 trở xuống (in ra sai số lớn nhất)
    Với kiến trúc không có BN (ViT, Swin, ConvNeXt dùng LayerNorm), mục này không áp dụng; ghi rõ.
    """
    fused_model = copy.deepcopy(model).eval()

    def fuse_children(module):
        children = list(module.named_children())
        previous_name, previous_child = None, None
        for name, child in children:
            if isinstance(previous_child, torch.nn.Conv2d) and isinstance(child, torch.nn.BatchNorm2d):
                fused_conv = torch.nn.utils.fusion.fuse_conv_bn_eval(previous_child, child)
                setattr(module, previous_name, fused_conv)
                setattr(module, name, torch.nn.Identity())
                previous_child = fused_conv
            else:
                fuse_children(child)
                previous_name, previous_child = name, child

    fuse_children(fused_model)
    return fused_model
