# DeepWeeds — Track 4 Day 2

**Học viên:** Nguyễn Mạnh Cường  
**Mã học viên:** 2A202602823

## Tổng quan

Bài làm phân loại DeepWeeds 9 lớp trên fold 0 được cung cấp. Pipeline bao gồm kiểm tra dữ liệu, smoke test, so sánh 5 backbone, ablation công thức huấn luyện, 8 cấu hình suy luận, benchmark latency và đánh giá chung kết qua 3 seed.

Cấu hình cuối `F01` dùng ConvNeXt-Tiny với công thức `T00` và single-view độ phân giải 256 (`I04`). Kết quả test đạt macro-F1 `0,97264 ± 0,00098` và top-1 `0,97776 ± 0,00057`.

## Notebook

- Notebook chạy chính: [`lab_day2.ipynb`](lab_day2.ipynb)
- Báo cáo: [`report.md`](report.md)
- Bảng kết quả: [`results.xlsx`](results.xlsx)

Notebook được chạy trên máy local. Khi nộp qua Colab/Kaggle, tải notebook lên nền tảng và thay `REPO_DIR`, `LABELS_DIR`, `IMAGES_DIR` nếu cấu trúc thư mục khác.

## Cấu trúc bài nộp

```text
.
├── README.md
├── report.md
├── results.xlsx
├── lab_day2.ipynb
├── code/
│   ├── dataset.py
│   ├── model.py
│   ├── losses.py
│   ├── train.py
│   ├── inference.py
│   ├── benchmark.py
│   ├── benchmark_backbones.py
│   ├── build_results_workbook.py
│   ├── eval.py
│   ├── generate_final_val.py
│   ├── self_test.py
│   └── make_report_assets.py
├── curves/
├── predictions/
├── report_assets/
├── eval_out/
└── runs/                 # chỉ config/history/summary trong gói nộp
```

Dataset và checkpoint `best.pt` không được đóng gói. Predictions test cần thiết để giảng viên tính lại chỉ số được giữ trong `predictions/`.

## Môi trường đã chạy

| Thành phần | Phiên bản |
|---|---|
| OS | Windows |
| Python | 3.11.16 |
| PyTorch | 2.14.0+cu130 |
| torchvision | 0.29.0+cu130 |
| timm | 1.0.30 |
| pandas | 3.0.6 |
| scikit-learn | 1.9.1 |
| openpyxl | 3.1.5 |
| Pillow | 12.3.0 |
| CUDA runtime | 13.0 |
| GPU | NVIDIA GeForce RTX 5060 Ti |

Các thư viện chính có thể cài bằng:

```bash
pip install torch torchvision timm pandas scikit-learn openpyxl pillow matplotlib
```

Nếu tải pretrained weights từ Hugging Face Hub nhiều lần, có thể đặt `HF_TOKEN` để tránh giới hạn request. Token không được lưu trong code hoặc bài nộp.

## Chuẩn bị dữ liệu

Đặt dữ liệu bên ngoài thư mục bài nộp theo cấu trúc:

```text
data/
├── images/
│   └── images/
│       └── *.jpg
└── labels/
    ├── labels.csv
    ├── train_subset0.csv
    ├── val_subset0.csv
    └── test_subset0.csv
```

Không sửa hoặc tự chia lại các file split. Train chỉ cập nhật trọng số, validation dùng để chọn cấu hình, test chỉ chạy sau khi đã khóa cấu hình.

## Thứ tự chạy

### 1. Chạy notebook

Mở `lab_day2.ipynb`, chọn kernel của môi trường trên và chạy lần lượt từ đầu đến cuối. Notebook tự tìm root repo từ vị trí hiện tại; nếu chạy trên nền tảng khác, đặt đường dẫn dữ liệu ở cell cấu hình đầu.

### 2. Chạy một thí nghiệm từ CLI

Từ thư mục chứa bài nộp, ví dụ chạy mốc:

```powershell
python code\train.py --set exp_id=T00 backbone=convnext_tiny seed=0 fold=0 images_dir=..\data\images\images labels_dir=..\data\labels out_dir=runs pred_dir=predictions num_workers=0
```

Ví dụ chạy cấu hình có thay đổi:

```powershell
python code\train.py --set exp_id=T04 backbone=convnext_tiny seed=0 lr_head=0.0001 images_dir=..\data\images\images labels_dir=..\data\labels num_workers=0
```

Trên Windows/Python 3.14+, nên để `num_workers=0` nếu gặp lỗi pickle worker. Phiên bản trong bài đã chuyển hàm seed worker ra module scope, nhưng `num_workers=0` vẫn là lựa chọn ổn định nhất để tái lập trên Windows.

### 3. Seed chung kết

Các seed đã dùng là:

```text
0, 1, 2
```

Với mỗi seed, huấn luyện `T00`, sau đó chạy:

- Baseline `I00`: single-view 224, lưu `T00_seed<k>_test.csv`.
- Final `I04`: single-view 256, lưu `F01_seed<k>_test.csv`.

Các file prediction có cột `Filename, y_true, y_pred, p0, ..., p8`, theo đúng thứ tự label 0–8.

### 4. Tính lại chỉ số

Từ thư mục bài nộp:

```powershell
python -X utf8 code\eval.py score --pred "predictions/F01_seed*_test.csv" --test-csv ..\data\labels\test_subset0.csv --labels ..\data\labels\labels.csv --tag F01 --out eval_out
python -X utf8 code\eval.py score --pred "predictions/T00_seed*_test.csv" --test-csv ..\data\labels\test_subset0.csv --labels ..\data\labels\labels.csv --tag T00 --out eval_out
```

Prediction validation của F01 cho ba seed được tạo bằng:

```powershell
python -X utf8 code\generate_final_val.py
```

Sau đó có thể tự chấm mục I, bao gồm kiểm tra val–test:

```powershell
python -X utf8 code\eval.py grade --final "predictions/F01_seed*_test.csv" --baseline "predictions/T00_seed*_test.csv" --final-val "predictions/F01_seed*_val.csv" --latency-p95-ms 3.77024 --latency-method proper --test-csv ..\data\labels\test_subset0.csv --val-csv ..\data\labels\val_subset0.csv --labels ..\data\labels\labels.csv --out eval_out
```

Kết quả kỳ vọng:

| Cấu hình | Macro-F1 mean ± std | Top-1 mean ± std | ECE mean |
|---|---:|---:|---:|
| F01 | 0,97264 ± 0,00098 | 0,97776 ± 0,00057 | 0,01320 |
| T00 | 0,96783 ± 0,00191 | 0,97434 ± 0,00151 | 0,01522 |

### 5. Tạo lại hình của báo cáo

```powershell
python code\make_report_assets.py
```

Script đọc các CSV/predictions đã lưu và ghi hình vào `report_assets/`.

Tạo lại bảng kết quả đầy đủ:

```powershell
python -X utf8 code\benchmark_backbones.py
python -X utf8 code\build_results_workbook.py
```

### 6. Chạy kiểm tra mã

```powershell
python -X utf8 code\self_test.py
```

Bốn kiểm tra bao gồm focal loss với `gamma=0`, CutMix, gộp Conv–BatchNorm và chuẩn hóa xác suất sau gộp view.

## Quy ước thí nghiệm

- `B01`–`B05`: so sánh backbone.
- `T00`: công thức huấn luyện nền.
- `T01`–`T05`: mỗi lần thay đúng một yếu tố.
- `T10`: kết hợp các yếu tố thử nghiệm.
- `I00`: inference mốc.
- `I01`–`I07`: TTA, multi-scale, ensemble, resolution, calibration và FP16.
- `F01`: cấu hình chung kết `T00 + I04`.

Mỗi lần train ghi `config.json`, `history.csv`, `summary.json`, curve và prediction validation. Checkpoint được chọn bằng macro-F1 validation.

## Ghi chú tái lập

- Latency được đo với 10 warmup và 100 lần lặp, có đồng bộ CUDA, batch 1; không gồm preprocessing.
- AMP được bật trong huấn luyện. Seed được cố định cho Python, NumPy và PyTorch; cuDNN deterministic được bật.
- Chênh lệch nhỏ hơn độ lệch chuẩn qua seed không được xem là cải thiện chắc chắn.
- Không dùng test để chọn backbone, hyperparameter, checkpoint, nhiệt độ hoặc inference.
