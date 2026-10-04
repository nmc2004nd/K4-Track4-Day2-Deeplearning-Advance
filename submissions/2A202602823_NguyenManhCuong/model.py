"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Giao diện bạn phải giữ:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
"""
from __future__ import annotations

import torch
import torch.nn as nn
import timm

# Gợi ý backbone (GUIDE.md mục 2.1). Tag trọng số của timm có thể đổi theo phiên bản:
# dùng timm.list_pretrained("resnet50*") để xem, và GHI LẠI tag bạn dùng trong results.xlsx.
SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",      # hoặc vit_small_patch16_224
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",        # mạng nhẹ
    "mobilenetv3": "mobilenetv3_large_100",      # mạng nhẹ
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    """Tạo model phân loại 9 lớp.

    `init` (trục A của GUIDE.md mục 3):
      - "scratch"  : pretrained=False, huấn luyện toàn bộ
      - "frozen"   : pretrained=True, đóng băng backbone, chỉ train head
      - "finetune" : pretrained=True, train toàn bộ

    TODO:
      - timm.create_model(name, pretrained=..., num_classes=num_classes, drop_rate=...)
        (timm tự thay head mới; head khởi tạo ngẫu nhiên)
      - nếu init == "frozen": gọi freeze_backbone(model)
      - ghi lại tên tag trọng số thực sự được tải (model.pretrained_cfg)
    """
    if init not in {"scratch", "frozen", "finetune"}:
        raise ValueError("init phải là 'scratch', 'frozen' hoặc 'finetune'")
    use_pretrained = bool(pretrained and init != "scratch")
    network = timm.create_model(
        name,
        pretrained=use_pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    cfg = getattr(network, "pretrained_cfg", {}) or {}
    network.weight_tag = (
        "random-init" if not use_pretrained
        else (cfg.get("hf_hub_id") or cfg.get("url") or name)
    )
    network._backbone_frozen = False
    if init == "frozen":
        freeze_backbone(network)
    return network


def freeze_backbone(model) -> None:
    """Đóng băng mọi tham số trừ head.

    TODO:
      - requires_grad = False cho tham số backbone; head (model.get_classifier()) vẫn train
      - lưu ý (GUIDE.md mục 3.2): backbone đóng băng thì BatchNorm cũng phải ở chế độ eval.
        Hãy nghĩ nơi nào trong train loop phải gọi lại model.train() mà vẫn giữ BN ở eval.
    """
    classifier = model.get_classifier()
    head_parameter_ids = {id(parameter) for parameter in classifier.parameters()}
    for parameter in model.parameters():
        parameter.requires_grad = id(parameter) in head_parameter_ids
    model._backbone_frozen = True

    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """Chia tham số thành 3 nhóm như slide Day 2, trang 52.

    - backbone có ndim > 1: lr = lr_backbone, weight_decay = weight_decay
    - norm và bias của backbone (ndim <= 1): lr = lr_backbone, weight_decay = 0
    - head mới: lr = lr_head (thường gấp 10 lần backbone), weight_decay = weight_decay

    TODO:
      - bỏ qua tham số requires_grad == False
      - trả về list[dict] dạng {"params": [...], "lr": ..., "weight_decay": ...}
      - (trục E) mở rộng: LR theo tầng nếu bạn muốn thử
    """
    classifier = model.get_classifier()
    head_parameter_ids = {id(parameter) for parameter in classifier.parameters()}
    backbone_decay, backbone_no_decay, head = [], [], []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in head_parameter_ids:
            head.append(parameter)
        elif parameter.ndim <= 1:
            backbone_no_decay.append(parameter)
        else:
            backbone_decay.append(parameter)

    groups = []
    if backbone_decay:
        groups.append({"params": backbone_decay, "lr": lr_backbone, "weight_decay": weight_decay})
    if backbone_no_decay:
        groups.append({"params": backbone_no_decay, "lr": lr_backbone, "weight_decay": 0.0})
    if head:
        groups.append({"params": head, "lr": lr_head, "weight_decay": weight_decay})
    if not groups:
        raise ValueError("Model không có tham số nào cần tối ưu")
    return groups


def count_params(model) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng. TODO."""
    return sum(parameter.numel() for parameter in model.parameters()) / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size (slide tính MAC, không phải FLOPs 2x).

    TODO: dùng thư viện đếm (fvcore, ptflops, thop...) hoặc tự đếm bằng hook.
    Ghi rõ công cụ đã dùng; số có thể lệch vài phần trăm giữa các công cụ.
    """
    macs = 0
    handles = []

    def conv_hook(module, inputs, output):
        nonlocal macs
        output_elements = output.numel()
        kernel_ops = module.kernel_size[0] * module.kernel_size[1]
        kernel_ops *= module.in_channels // module.groups
        macs += output_elements * kernel_ops

    def linear_hook(module, inputs, output):
        nonlocal macs
        macs += output.numel() * module.in_features

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))

    parameter = next(model.parameters())
    device = parameter.device
    was_training = model.training
    try:
        model.eval()
        with torch.inference_mode():
            model(torch.zeros(1, 3, img_size, img_size, device=device))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return macs / 1e9
