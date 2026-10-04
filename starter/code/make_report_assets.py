"""Tạo các hình dùng trong report.md từ log và prediction đã lưu."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


STARTER_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = STARTER_DIR.parent
LABELS_DIR = REPO_DIR / "data" / "labels"
IMAGES_DIR = REPO_DIR / "data" / "images" / "images"
OUT_DIR = STARTER_DIR / "report_assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "Chinee apple", "Lantana", "Parkinsonia", "Parthenium",
    "Prickly acacia", "Rubber vine", "Siam weed", "Snake weed", "Negative",
]


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_split_distribution() -> None:
    rows = []
    for split in ("train", "val", "test"):
        frame = pd.read_csv(LABELS_DIR / f"{split}_subset0.csv")
        counts = frame["Label"].value_counts().reindex(range(9), fill_value=0)
        rows.append(counts.to_numpy())
    values = np.asarray(rows)
    x = np.arange(9)
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for index, split in enumerate(("Train", "Val", "Test")):
        ax.bar(x + (index - 1) * width, values[index], width, label=split)
    ax.set_xticks(x, CLASS_NAMES, rotation=30, ha="right")
    ax.set_ylabel("Số ảnh")
    ax.set_title("Phân bố lớp của fold 0")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save(fig, "class_distribution.png")


def plot_backbones() -> None:
    frame = pd.read_csv(STARTER_DIR / "backbone_results.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.scatter(frame["#params (M)"], frame["macro-F1 val"], s=75, color="#2563eb")
    for _, row in frame.iterrows():
        ax.annotate(row["exp_id"], (row["#params (M)"], row["macro-F1 val"]),
                    xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Số tham số (triệu)")
    ax.set_ylabel("Macro-F1 validation")
    ax.set_title("Độ chính xác và kích thước backbone")
    ax.grid(alpha=0.25)
    save(fig, "backbone_tradeoff.png")


def plot_inference_tradeoff() -> None:
    frame = pd.read_csv(STARTER_DIR / "inference_results.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#dc2626" if exp == "I04" else "#0f766e" for exp in frame["exp_id"]]
    ax.scatter(frame["p95_ms"], frame["macro_f1_val"], s=80, c=colors)
    offsets = {"I00": (6, -15), "I06": (6, 8), "I07": (6, 21)}
    for row in frame.itertuples(index=False):
        ax.annotate(row.exp_id, (row.p95_ms, row.macro_f1_val),
                    xytext=offsets.get(row.exp_id, (5, 4)), textcoords="offset points")
    ax.set_xlabel("Độ trễ p95, batch 1 (ms)")
    ax.set_ylabel("Macro-F1 validation")
    ax.set_title("Đánh đổi độ chính xác – độ trễ")
    ax.grid(alpha=0.25)
    save(fig, "inference_tradeoff.png")


def plot_confusion() -> None:
    matrix = pd.read_csv(STARTER_DIR / "eval_out" / "F01_confusion_sum.csv", index_col=0).to_numpy()
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Tỷ lệ theo lớp thật")
    ax.set_xticks(range(9), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(9), CLASS_NAMES)
    ax.set_xlabel("Nhãn dự đoán")
    ax.set_ylabel("Nhãn thật")
    ax.set_title("Confusion matrix chuẩn hóa của F01 (gộp 3 seed)")
    for i in range(9):
        for j in range(9):
            value = normalized[i, j]
            if value >= 0.01 or i == j:
                ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if value > 0.55 else "black")
    save(fig, "f01_confusion_matrix.png")


def plot_errors() -> None:
    frame = pd.read_csv(STARTER_DIR / "predictions" / "F01_seed0_test.csv")
    errors = frame[frame["y_true"] != frame["y_pred"]].copy()
    prob_cols = [f"p{i}" for i in range(9)]
    errors["confidence"] = errors[prob_cols].max(axis=1)
    chosen = (
        errors.sort_values("confidence", ascending=False)
        .drop_duplicates("y_true")
        .head(8)
    )
    fig, axes = plt.subplots(2, 4, figsize=(14, 8.5))
    for ax, (_, row) in zip(axes.flat, chosen.iterrows()):
        path = IMAGES_DIR / row["Filename"]
        ax.imshow(Image.open(path).convert("RGB"))
        true_name = CLASS_NAMES[int(row["y_true"])]
        pred_name = CLASS_NAMES[int(row["y_pred"])]
        ax.set_title(f"Thật: {true_name}\nĐoán: {pred_name} ({row['confidence']:.2f})",
                     fontsize=9, pad=6)
        ax.axis("off")
    for ax in axes.flat[len(chosen):]:
        ax.axis("off")
    fig.suptitle("Ví dụ F01 dự đoán sai trên test, seed 0", fontsize=13, y=0.98)
    fig.subplots_adjust(top=0.88, hspace=0.45, wspace=0.08)
    fig.savefig(OUT_DIR / "f01_error_examples.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-whitegrid")
    plot_split_distribution()
    plot_backbones()
    plot_inference_tradeoff()
    plot_confusion()
    plot_errors()
    print(f"Report assets written to {OUT_DIR}")
