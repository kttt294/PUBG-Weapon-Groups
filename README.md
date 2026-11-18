# PUBG Weapon Groups

Dự án Khai phá dữ liệu phân loại nhóm vũ khí trong PUBG từ ngữ cảnh của sự kiện hạ gục và lịch sử hạ gục trước đó trong cùng trận.

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
