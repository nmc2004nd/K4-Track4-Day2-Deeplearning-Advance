"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

Mã hoàn thiện dùng MỘT hàm `run(cfg)` cho mọi cấu hình
(RUBRIC mục H): đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
Chỉ số dùng để chọn checkpoint (macro-F1 val) phải tính bằng eval.compute_metrics của repo gốc,
để cùng định nghĩa với lúc chấm:
    sys.path.insert(0, "<thư mục chứa eval.py>");  from eval import compute_metrics
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import copy
import json
import math
import random
import time
from types import UnionType
from typing import get_args, get_origin, get_type_hints

import numpy as np
import pandas as pd
import torch

# Ghi file dự đoán đúng định dạng bằng hàm có sẵn trong eval.py (repo gốc):
#     from eval import save_predictions, compute_metrics
# Log theo epoch (history.csv) và config.json bạn tự ghi bằng pandas/json.


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug ...
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên.

    TODO: random, numpy, torch (CPU và CUDA); cân nhắc cudnn.deterministic/benchmark và
    seed cho worker của DataLoader. Ghi lại trong báo cáo mức độ tái lập bạn đạt được.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model, cfg: Config):
    """AdamW với 3 nhóm tham số (xem model.param_groups). TODO."""
    import model as model_utils

    groups = model_utils.param_groups(
        model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay
    )
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về ~0 (slide trang 55). TODO.

    Cập nhật theo bước (iteration) hoặc theo epoch đều được; ghi rõ bạn chọn gì.
    Gợi ý kiểm tra: vẽ đường LR theo bước để thấy đúng hình warmup + cosine.
    """
    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch phải là số dương")
    total_steps = max(1, cfg.epochs * steps_per_epoch)
    warmup_steps = max(0, round(cfg.warmup_epochs * steps_per_epoch))

    def lr_multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W  (slide trang 56).

    TODO:
      - __init__(model, decay): sao chép trọng số
      - update(model): sau mỗi bước tối ưu
      - copy_to(model) hoặc dùng bản sao riêng để đánh giá bằng trọng số EMA
      - lưu ý BatchNorm: buffer (running_mean/var) cũng phải được xử lý hợp lý
    """

    def __init__(self, model, decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay phải nằm trong (0, 1)")
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad = False

    def update(self, model) -> None:
        with torch.no_grad():
            source = model.state_dict()
            for name, value in self.module.state_dict().items():
                new_value = source[name].detach()
                if value.is_floating_point():
                    value.mul_(self.decay).add_(new_value, alpha=1.0 - self.decay)
                else:
                    value.copy_(new_value)

    def copy_to(self, model) -> None:
        model.load_state_dict(self.module.state_dict())


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    """Một epoch huấn luyện. Trả về dict, ví dụ {"train_loss": ..., "lr": ...}.

    TODO:
      - model.train() (nếu init == "frozen": giữ phần backbone ở eval, xem model.freeze_backbone)
      - nếu cfg.mix: mix_batch rồi mixed_loss (losses.py)
      - AMP (autocast + GradScaler), clip gradient nếu cần, optimizer.step(), scheduler.step()
      - nếu có EMA: ema.update(model)
    """
    import losses

    model.train()
    if getattr(model, "_backbone_frozen", False):
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()

    total_loss, total_items = 0.0, 0
    amp_enabled = bool(cfg.amp and device.type == "cuda")
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        targets = labels
        if cfg.mix:
            images, targets = losses.mix_batch(images, labels, cfg.mix_alpha, cfg.mix)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = (
                losses.mixed_loss(criterion, logits, targets)
                if cfg.mix else criterion(logits, labels)
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if ema is not None:
            ema.update(model)

        batch_size = images.size(0)
        total_loss += loss.detach().item() * batch_size
        total_items += batch_size

    return {
        "train_loss": total_loss / max(1, total_items),
        "lr": float(optimizer.param_groups[0]["lr"]),
    }


def evaluate(model, loader, criterion, device):
    """Chạy model trên một loader ở chế độ eval, KHÔNG tính gradient.

    Trả về (filenames: list[str], y_true: ndarray[N], logits: ndarray[N, 9], loss: float).
    Giữ đúng thứ tự của loader để ghép logit với tên file.

    TODO: model.eval(), torch.inference_mode(), gom kết quả. Softmax khi cần xác suất.
    """
    model.eval()
    filenames, labels_all, logits_all = [], [], []
    total_loss, total_items = 0.0, 0
    with torch.inference_mode():
        for images, labels, batch_filenames in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_items += batch_size
            filenames.extend(list(batch_filenames))
            labels_all.append(labels.cpu())
            logits_all.append(logits.float().cpu())
    return (
        filenames,
        torch.cat(labels_all).numpy(),
        torch.cat(logits_all).numpy(),
        total_loss / max(1, total_items),
    )


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    """Vẽ đường cong training của một thí nghiệm -> curves/<exp_id>_<mota>.png (GUIDE.md mục 6.2).

    TODO: tối thiểu loss train/val và macro-F1 val theo epoch; có tiêu đề, nhãn trục, chú thích;
    khuyến khích thêm LR theo bước. Lưu bằng matplotlib với dpi đủ nét để đọc số.
    """
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(history)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(frame["epoch"], frame["train_loss"], label="train")
    axes[0].plot(frame["epoch"], frame["val_loss"], label="val")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(frame["epoch"], frame["val_macro_f1"], label="macro-F1")
    axes[1].plot(frame["epoch"], frame["val_top1"], label="top-1")
    axes[1].set(xlabel="Epoch", ylabel="Score", title="Validation metrics", ylim=(0, 1))
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết. Trả về dict kết quả tóm tắt.

    TODO theo thứ tự:
      1. set_seed; tạo thư mục run_dir(cfg); ghi config.json (dataclasses.asdict(cfg))
      2. dataset.load_split + dataset.check_split (dừng nếu vi phạm S1-S6)
      3. dựng train/val loader (test loader chỉ tạo khi cfg.save_test_predictions)
      4. model.build_model, criterion (losses.build_criterion), optimizer, scheduler, scaler, EMA
      5. với mỗi epoch: train_one_epoch -> evaluate(val) -> ghi history (loss, macro-F1 val, lr...)
         và lưu checkpoint tốt nhất theo MACRO-F1 VAL (hòa thì lấy epoch sớm hơn)
      6. cuối: nạp checkpoint tốt nhất, lưu val logits và eval.save_predictions(pred_path(cfg, "val"), ...)
      7. NẾU cfg.save_test_predictions (chỉ ở Bước 4): đánh giá test đúng MỘT lần,
         lưu logits và eval.save_predictions(pred_path(cfg, "test"), ...)
      8. ghi history.csv, plot_curves(...), trả về dict tóm tắt
         (best_epoch, macro-F1 val, thời gian train mỗi epoch, số tham số, GMAC)
    Quy tắc: KHÔNG dùng test để chọn checkpoint hay bất kỳ quyết định nào (README.md, S4).
    """
    import dataset
    import losses
    import model as model_utils

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(repo_root))
    from eval import compute_metrics, save_predictions

    if cfg.epochs <= 0:
        raise ValueError("epochs phải là số dương")
    set_seed(cfg.seed)
    output_dir = run_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    train_df, val_df, test_df = dataset.load_split(cfg.labels_dir, cfg.fold)
    dataset.check_split(train_df, val_df, test_df, cfg.images_dir)
    train_transform = dataset.build_transforms(True, cfg.img_size, cfg.aug)
    eval_transform = dataset.build_transforms(False, cfg.img_size, cfg.aug)
    train_loader = dataset.make_loader(
        train_df, cfg.images_dir, train_transform, cfg.batch_size,
        train=True, sampler=cfg.sampler, num_workers=cfg.num_workers,
    )
    val_loader = dataset.make_loader(
        val_df, cfg.images_dir, eval_transform, cfg.batch_size,
        train=False, num_workers=cfg.num_workers,
    )
    test_loader = None
    if cfg.save_test_predictions:
        test_loader = dataset.make_loader(
            test_df, cfg.images_dir, eval_transform, cfg.batch_size,
            train=False, num_workers=cfg.num_workers,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = model_utils.build_model(
        cfg.backbone,
        pretrained=cfg.init != "scratch",
        num_classes=dataset.NUM_CLASSES,
        drop_rate=cfg.drop_rate,
        init=cfg.init,
    ).to(device)
    params_m = model_utils.count_params(network)
    gmac = model_utils.count_gmacs(network, cfg.img_size)
    weight_tag = str(getattr(network, "weight_tag", cfg.backbone))

    weight = None
    if cfg.loss == "ce_weighted":
        counts = train_df["Label"].value_counts().reindex(range(dataset.NUM_CLASSES)).to_numpy()
        weight = losses.class_weights(counts, cfg.class_weight_beta or 0.0).to(device)
    criterion = losses.build_criterion(
        cfg.loss,
        smoothing=cfg.label_smoothing,
        gamma=cfg.focal_gamma,
        weight=weight,
    ).to(device)
    optimizer = build_optimizer(network, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    amp_enabled = bool(cfg.amp and device.type == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    ema = EMA(network, cfg.ema_decay) if cfg.ema_decay is not None else None

    history = []
    best_macro_f1 = -1.0
    best_epoch = 0
    checkpoint_path = output_dir / "best.pt"
    for epoch in range(1, cfg.epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        train_stats = train_one_epoch(
            network, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_seconds = time.perf_counter() - start

        eval_network = ema.module if ema is not None else network
        _, y_true, logits, val_loss = evaluate(eval_network, val_loader, criterion, device)
        probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        metrics = compute_metrics(y_true, probs.argmax(1), probs)
        row = {
            "epoch": epoch,
            **train_stats,
            "val_loss": val_loss,
            "val_macro_f1": metrics["macro_f1"],
            "val_top1": metrics["top1"],
            "epoch_seconds": epoch_seconds,
        }
        history.append(row)
        print(
            f"[{cfg.exp_id}] epoch {epoch:02d}/{cfg.epochs} "
            f"loss={row['train_loss']:.4f} val_f1={row['val_macro_f1']:.4f} "
            f"val_top1={row['val_top1']:.4f} time={epoch_seconds:.1f}s"
        )
        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_epoch = epoch
            torch.save(eval_network.state_dict(), checkpoint_path)

    try:
        best_state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        best_state = torch.load(checkpoint_path, map_location=device)
    network.load_state_dict(best_state)
    val_filenames, val_true, val_logits, _ = evaluate(network, val_loader, criterion, device)
    val_probs = torch.softmax(torch.from_numpy(val_logits), dim=1).numpy()
    val_metrics = compute_metrics(val_true, val_probs.argmax(1), val_probs)
    save_predictions(pred_path(cfg, "val"), val_filenames, val_true, val_probs)

    if test_loader is not None:
        test_filenames, test_true, test_logits, _ = evaluate(network, test_loader, criterion, device)
        test_probs = torch.softmax(torch.from_numpy(test_logits), dim=1).numpy()
        save_predictions(pred_path(cfg, "test"), test_filenames, test_true, test_probs)

    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    curve_path = Path("curves") / f"{cfg.exp_id}_seed{cfg.seed}.png"
    plot_curves(history, curve_path, f"{cfg.exp_id} — {cfg.backbone}")
    result = {
        "exp_id": cfg.exp_id,
        "backbone": cfg.backbone,
        "weight_tag": weight_tag,
        "params_m": params_m,
        "gmac": gmac,
        "best_epoch": best_epoch,
        "val_macro_f1": val_metrics["macro_f1"],
        "val_top1": val_metrics["top1"],
        "seconds_per_epoch": float(np.mean([row["epoch_seconds"] for row in history])),
        "seed": cfg.seed,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def parse_overrides(pairs: list[str]) -> dict:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict, ép kiểu theo field của Config.

    TODO: tách key/value, báo lỗi rõ nếu key không có trong Config, ép int/float/bool/None theo kiểu field.
    """
    config_fields = {field.name: field for field in fields(Config)}
    type_hints = get_type_hints(Config)
    parsed: dict[str, object] = {}

    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Override phải có dạng KEY=VALUE, nhận được: {pair!r}")
        key, raw_value = pair.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not key or key not in config_fields:
            valid = ", ".join(sorted(config_fields))
            raise KeyError(f"Config không có trường {key!r}. Các trường hợp lệ: {valid}")

        annotation = type_hints[key]
        origin = get_origin(annotation)
        candidates = list(get_args(annotation)) if origin in (UnionType, getattr(__import__('typing'), 'Union')) else [annotation]
        allows_none = type(None) in candidates
        candidates = [candidate for candidate in candidates if candidate is not type(None)]

        if raw_value.lower() in {"none", "null"}:
            if not allows_none:
                raise ValueError(f"{key} không cho phép None")
            parsed[key] = None
            continue

        target = candidates[0] if candidates else str
        try:
            if target is bool:
                lowered = raw_value.lower()
                if lowered in {"true", "1", "yes", "on"}:
                    value = True
                elif lowered in {"false", "0", "no", "off"}:
                    value = False
                else:
                    raise ValueError("giá trị bool phải là true/false")
            elif target is int:
                value = int(raw_value)
            elif target is float:
                value = float(raw_value)
            else:
                value = raw_value
        except ValueError as exc:
            raise ValueError(f"Không thể ép {key}={raw_value!r} sang {target}") from exc
        parsed[key] = value

    return parsed


def main() -> None:
    """Điểm vào dòng lệnh: `python train.py --set exp_id=B01 backbone=resnet50 seed=0`.

    TODO: argparse nhận `--set KEY=VALUE ...`, dựng Config qua parse_overrides, gọi run(cfg), in kết quả.
    """
    parser = argparse.ArgumentParser(description="Huấn luyện một cấu hình DeepWeeds")
    parser.add_argument(
        "--set",
        metavar="KEY=VALUE",
        nargs="*",
        default=[],
        help="Ghi đè trường Config, ví dụ --set exp_id=B01 backbone=resnet50 seed=0",
    )
    args = parser.parse_args()
    cfg = Config(**parse_overrides(args.set))
    result = run(cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
