"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Mã hoàn thiện cho dữ liệu, biến đổi ảnh và DataLoader.
Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1. Đọc trước khi viết.

Giao diện bạn phải giữ (để notebook, train.py và eval.py ghép được với nhau):
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)
"""
from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # đổi nếu trọng số timm bạn dùng yêu cầu mean/std khác
IMAGENET_STD = (0.229, 0.224, 0.225)


def _resolve_project_path(path: str | Path) -> Path:
    """Resolve a relative path from either the current directory or repository root."""
    path = Path(path).expanduser()
    if path.is_absolute() or path.exists():
        return path

    repo_path = Path(__file__).resolve().parent.parent / path
    return repo_path if repo_path.exists() else path


def _seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker; top-level so it remains picklable on Windows."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).

    Mỗi file có cột `Filename, Label, Species`. Trả về ba DataFrame.
    KHÔNG sửa, lọc hay chia lại dữ liệu.

    TODO:
      - đọc ba file CSV bằng pandas
      - trả về (train_df, val_df, test_df)
    """
    labels_dir = _resolve_project_path(labels_dir)
    if not isinstance(fold, int) or fold < 0:
        raise ValueError(f"fold phải là số nguyên không âm, nhận được: {fold!r}")

    dataframes = []
    required_columns = {"Filename", "Label"}
    for split_name in ("train", "val", "test"):
        csv_path = labels_dir / f"{split_name}_subset{fold}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file split: {csv_path}")

        df = pd.read_csv(csv_path)
        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            raise ValueError(
                f"{csv_path} thiếu cột bắt buộc: {sorted(missing_columns)}"
            )
        dataframes.append(df)

    return tuple(dataframes)


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu.

    TODO kiểm tra, mỗi ý lỗi thì `assert` / raise để dừng ngay:
      1. số ảnh mỗi tập và số ảnh mỗi lớp trong từng tập (kỳ vọng xấp xỉ 60/20/20)
      2. giao của từng cặp tập theo Filename phải RỖNG (train∩val, train∩test, val∩test)
      3. hợp ba tập phải bằng đúng 17.509 ảnh
      4. mọi Filename đều tồn tại trong `images_dir`
    Trả về dict, ví dụ {"n": {...}, "per_class": {...}, "overlap": {...}} để dán vào báo cáo.
    """
    splits = {"train": train_df, "val": val_df, "test": test_df}
    required_columns = {"Filename", "Label"}

    for split_name, df in splits.items():
        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            raise ValueError(
                f"Split {split_name} thiếu cột: {sorted(missing_columns)}"
            )
        if df[list(required_columns)].isna().any().any():
            raise ValueError(f"Split {split_name} chứa Filename/Label rỗng")
        if df["Filename"].duplicated().any():
            duplicates = df.loc[df["Filename"].duplicated(), "Filename"].tolist()
            raise ValueError(
                f"Split {split_name} chứa Filename trùng, ví dụ: {duplicates[:5]}"
            )

        labels = pd.to_numeric(df["Label"], errors="coerce")
        if labels.isna().any() or not labels.between(0, NUM_CLASSES - 1).all():
            raise ValueError(
                f"Split {split_name} có Label không hợp lệ; cần nằm trong [0, {NUM_CLASSES - 1}]"
            )

    filename_sets = {
        split_name: set(df["Filename"].astype(str))
        for split_name, df in splits.items()
    }
    overlap = {
        "train_val": len(filename_sets["train"] & filename_sets["val"]),
        "train_test": len(filename_sets["train"] & filename_sets["test"]),
        "val_test": len(filename_sets["val"] & filename_sets["test"]),
    }
    if any(overlap.values()):
        raise ValueError(f"Các split bị trùng Filename: {overlap}")

    all_filenames = set().union(*filename_sets.values())
    if len(all_filenames) != 17_509:
        raise ValueError(
            f"Hợp ba split phải có 17,509 ảnh, hiện có {len(all_filenames):,}"
        )

    images_dir = _resolve_project_path(images_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {images_dir}")
    missing_files = sorted(
        filename for filename in all_filenames
        if not (images_dir / filename).is_file()
    )
    if missing_files:
        raise FileNotFoundError(
            f"Thiếu {len(missing_files):,} ảnh trong {images_dir}; "
            f"ví dụ: {missing_files[:5]}"
        )

    n = {name: int(len(df)) for name, df in splits.items()}
    n["total"] = sum(n.values())
    per_class = {}
    for split_name, df in splits.items():
        counts = df["Label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0)
        per_class[split_name] = {
            CLASS_NAMES[label]: int(counts.loc[label])
            for label in range(NUM_CLASSES)
        }

    report = {
        "n": n,
        "per_class": per_class,
        "overlap": overlap,
        "missing_files": 0,
    }

    print("Split check: PASS")
    print("Image counts:", n)
    print("Pairwise overlap:", overlap)
    print("All 17,509 image files exist.")
    return report


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Tạo transform. `aug` chọn mức augmentation; bạn tự định nghĩa các giá trị.

    Gợi ý các giá trị `aug` (trục B của GUIDE.md mục 3): "basic", "color", "trivial", "randaug".
    Mixup/CutMix trộn theo batch nên nằm ở losses.py, không ở đây.

    Train (basic): RandomResizedCrop(img_size) + lật ngang + ToTensor + Normalize.
    Val/test: ảnh gốc 256x256 -> CenterCrop(img_size) (hoặc giữ nguyên 256; ghi rõ bạn chọn gì)
              + ToTensor + Normalize. KHÔNG augmentation ngẫu nhiên khi đánh giá.

    TODO: dùng torchvision.transforms (hoặc v2). Lưu ý: lật dọc có hợp lệ với ảnh cỏ dại không?
    """
    from torchvision import transforms

    if img_size <= 0:
        raise ValueError("img_size phải là số dương")

    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if not train:
        resize_size = round(img_size / 0.875)
        return transforms.Compose([
            transforms.Resize(resize_size, antialias=True),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ])

    if aug not in {"basic", "color", "trivial", "randaug"}:
        raise ValueError(
            f"aug không hợp lệ: {aug!r}; chọn basic, color, trivial hoặc randaug"
        )

    operations = [
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), antialias=True),
        transforms.RandomHorizontalFlip(),
    ]
    if aug == "color":
        operations.append(
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05)
        )
    elif aug == "trivial":
        operations.append(transforms.TrivialAugmentWide())
    elif aug == "randaug":
        operations.append(transforms.RandAugment(num_ops=2, magnitude=9))

    operations.extend([transforms.ToTensor(), normalize])
    return transforms.Compose(operations)


class DeepWeedsDataset(Dataset):  # TODO: kế thừa torch.utils.data.Dataset
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label).

    __getitem__(i) phải trả về (ảnh đã transform, nhãn int, tên file str).
    Tên file cần có để ghi `predictions/*.csv` đúng định dạng của eval.py.

    TODO:
      - __init__(self, df, images_dir, transform): giữ df, mở ảnh bằng PIL, chuyển sang RGB
      - __len__
      - __getitem__ -> (tensor, int(label), filename)
      - (tuỳ chọn) nạp trước ảnh vào RAM nếu bị nghẽn đọc đĩa trên Colab
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        required_columns = {"Filename", "Label"}
        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            raise ValueError(f"DataFrame thiếu cột: {sorted(missing_columns)}")

        self.df = df.reset_index(drop=True).copy()
        self.images_dir = _resolve_project_path(images_dir)
        self.transform = transform
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {self.images_dir}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        filename = str(row["Filename"])
        image_path = self.images_dir / filename
        if not image_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            else:
                image = image.copy()
        return image, int(row["Label"]), filename


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2):
    """Tạo DataLoader.

    TODO:
      - train=True: shuffle (hoặc dùng sampler); train=False: không shuffle, giữ thứ tự df
        (thứ tự phải ổn định để ghép logit với Filename)
      - sampler=None | "balanced": "balanced" dùng WeightedRandomSampler với trọng số
        1/(số ảnh của lớp) (trục D của GUIDE.md mục 3)
      - drop_last=True khi train nếu batch cuối quá nhỏ làm BatchNorm không ổn định
      - pin_memory=True, num_workers hợp lý; seed cho worker (worker_init_fn) để tái lập
    """
    if batch_size <= 0:
        raise ValueError("batch_size phải là số dương")
    if num_workers < 0:
        raise ValueError("num_workers không được âm")
    if sampler not in {None, "balanced"}:
        raise ValueError("sampler chỉ nhận None hoặc 'balanced'")
    if sampler == "balanced" and not train:
        raise ValueError("Balanced sampler chỉ được dùng cho tập train")

    dataset = DeepWeedsDataset(df, images_dir, transform)
    torch_sampler = None
    if sampler == "balanced":
        labels = df["Label"].astype(int).to_numpy()
        counts = np.bincount(labels, minlength=NUM_CLASSES)
        if (counts == 0).any():
            raise ValueError("Không thể tạo balanced sampler khi có lớp không có mẫu")
        sample_weights = torch.as_tensor(1.0 / counts[labels], dtype=torch.double)
        torch_sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True
        )

    generator = torch.Generator()
    generator.manual_seed(torch.initial_seed())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train and torch_sampler is None,
        sampler=torch_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
        worker_init_fn=_seed_worker,
        generator=generator,
        persistent_workers=num_workers > 0,
    )
