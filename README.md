# PUBG Weapon Groups

Dự án Khai phá dữ liệu phân loại nhóm vũ khí trong PUBG từ ngữ cảnh của sự kiện hạ gục và lịch sử đã quan sát trong trận. GitHub dự án https://github.com/kttt294/PUBG-Weapon-Groups.

## Notebook chính và giao thức hiện tại

Mở `PUBG_Weapon_Groups.ipynb` trên Kaggle, gắn dataset `skihikingkevin/pubg-match-deaths`, bật Internet và GPU T4 hoặc P100 rồi Run All. Notebook tải đúng commit và kiểm tra TabM, TabICLv2 trước khi quét dữ liệu. API key chỉ dùng trên máy gửi notebook lên Kaggle.

Notebook chính gọi `python -m pubg_weapon.study` với năm tệp CSV. Quy trình xử lý toàn bộ nguồn và học trên mọi dòng train hợp lệ, với ba seed mô hình 42, 2026, 7 và một cách chia dữ liệu cố định. Các tham số có trong `configs/study.json`.

```bash
python -m pip install -r requirements.txt -r requirements-modern.txt
python -m pubg_weapon.study --source /path/to/deaths/kill_match_stats_final_{0,1,2,3,4}.csv --output artifacts/study
```

Lượt chính khảo sát alpha và epoch của Logistic SGD, hai mức chiều sâu và trọng số của XGBoost. Chọn cấu hình bằng validation, lưu đường học và đánh giá test theo trận. Các kết quả từng seed cùng trung bình, độ lệch chuẩn và bootstrap được giữ trong đầu ra.

TabM (ICLR 2025) và TabICLv2 (ICML 2026) cùng sáu đối chứng chạy trên một bộ dữ liệu chung gồm nguyên các trận, tối đa 40.000 dòng mỗi tập. Mọi phương pháp và seed trong bộ so sánh dùng cùng hàng, thứ tự hàng và phân bố tự nhiên. Các mô hình toàn train dùng mọi dòng hợp lệ của phần train nguồn. Bảng điểm ở hai quy mô cần diễn giải theo dữ liệu học tương ứng.

TabM dùng `tabm==0.0.3`, 16 thành viên, 3 khối, chiều rộng 256, không có feature embeddings. TabICLv2 dùng `tabicl==2.2.0`, checkpoint `tabicl-classifier-v2-20260212.ckpt`, 4 thành viên và representation cache. Chi phí tiền huấn luyện bên ngoài PUBG không nằm trong thời gian fit đo tại đây.

Tập holdout đã được đánh giá trong lượt trước và được giữ cố định cho đánh giá lại theo giao thức này. Mọi lựa chọn tham số tiếp tục dựa vào validation. Độ lệch chuẩn ba seed phản ánh biến thiên của mô hình, còn khoảng tin cậy tính riêng bằng bootstrap theo trận.

## Sản phẩm đầu ra

| Đường dẫn trong kết quả | Nội dung |
|---|---|
| `prepared/` | Kiểm kê nguồn và định nghĩa chia tập, các CSV/JSON được xuất cùng gói |
| `eda/` | Thống kê toàn train, chất lượng từng cột và so sánh worker |
| `full_models/<seed>/` | Mô hình toàn train, learning curve, chọn cấu hình và test |
| `modern_comparison/` | Hai phương pháp mới, sáu đối chứng, dự đoán và thống kê qua ba seed |
| `comparison_cohorts/` | Dữ liệu dùng chung và SHA-256 từng bộ |
| `diagnostics/<seed>/` | Phân tích cự ly, nhóm nhãn và permutation importance |
| `deliverables/` | Báo cáo PDF 12,5 pt, nguồn Markdown/LaTeX và PowerPoint |
| `source.zip`, `commit.txt`, `environment.json` | Mã nguồn thực tế, commit và môi trường |

Chỉ khi tất cả công đoạn hoàn tất, `status.json` mới ghi `completed`. Khi lỗi, notebook vẫn giữ kết quả đã hoàn tất cùng thông báo để chẩn đoán. Mã nguồn LaTeX dùng XeLaTeX. Cỡ chữ 12,5 áp dụng cho chữ trong báo cáo PDF, kể cả bảng, bìa, số trang và mục lục.

Tài liệu chính thức cho [TabM](https://github.com/yandex-research/tabm), [TabICLv2](https://proceedings.mlr.press/v306/qu26a.html) và [API TabICL](https://tabicl.readthedocs.io/en/latest/api.html).

## Các đường chạy của thực nghiệm trước

Những lệnh bên dưới giữ các quy trình trước. Notebook chính ở trên thực hiện toàn bộ giao thức ba seed cùng hai phương pháp gần đây.

## Chạy thực nghiệm

Lượt chính thức trên bộ đầy đủ dùng năm tệp CSV sự kiện tử vong trong thư mục `deaths/`. Mọi dòng thuộc cùng trận được đưa vào cùng phân vùng, kể cả khi xuất hiện ở nhiều tệp. Pipeline xử lý toàn bộ các phân vùng và dùng mọi dòng hợp lệ của tập train. Tập validation chọn mô hình, tập test đánh giá sau khi lựa chọn đã khóa.

```bash
python -m pip install -r requirements.txt
python scripts/08_full_data.py --source /path/to/deaths/kill_match_stats_final_{0,1,2,3,4}.csv --output artifacts/full --jobs 2
```

Lượt đầy đủ dùng Gaussian Naive Bayes và SGD Logistic theo lô, cùng XGBoost Hist với bộ nhớ ngoài trên CPU. Baseline dự đoán lớp phổ biến nhất. Kết quả nhóm đặc trưng A–D dùng cùng cấu hình SGD. Nhãn ngoài bảng ánh xạ được đếm trong `unmapped_causes.csv` và loại khỏi tập có nhãn. Tệp `dataset_manifest.json` ghi số dòng từng tệp, số dòng theo split và thời gian tiền xử lý.

### Quy trình lấy mẫu dùng trong thực nghiệm trước

Dùng Python 3.11 trở lên trên Linux hoặc Kaggle. Tải [PUBG Match Deaths and Statistics](https://www.kaggle.com/datasets/skihikingkevin/pubg-match-deaths) và lấy tệp `kill_match_stats_final_0.csv` trong thư mục `deaths`. Trên Kaggle, gắn bộ dữ liệu vào notebook và truyền đường dẫn tệp trong `/kaggle/input` cho `--source`.

```bash
python -m pip install -r requirements.txt
python scripts/01_eda.py --source /path/to/kill_match_stats_final_0.csv --output data/interim/run01 --matches 3000
python scripts/02_preprocess.py --eda data/interim/run01 --output data/processed/run01 --max-train-rows 30000
python scripts/03_models.py --processed data/processed/run01 --output artifacts/run01 --jobs 2 --quick
python scripts/05_diagnostics.py --processed data/processed/run01 --run artifacts/run01 --output reports/run01-validation
```

Mỗi lượt chạy dùng một tên thư mục đầu ra mới. Các script tự tạo thư mục đầu ra và thư mục cha khi cần. Để huấn luyện đầy đủ, đặt `--matches 50000`, `--max-train-rows 1000000` và bỏ `--quick`.

Sau khi khóa lựa chọn mô hình theo Macro-F1 trên validation, có thể chạy đánh giá tập kiểm tra một lần cho lượt huấn luyện đó.

```bash
python scripts/04_evaluation.py --processed data/processed/run01 --run artifacts/run01 --output reports/run01-test --jobs 2
```

Notebook gọi trực tiếp các script từ `01_eda.py` đến `04_evaluation.py`. `05_diagnostics.py` phân tích validation. `06_rf_regularization.py` và `07_appendix.py` chạy các thí nghiệm bổ sung trong báo cáo.
