# PUBG Weapon Groups: Phân loại Nhóm Vũ khí từ Ngữ cảnh và Lịch sử Sự kiện

Dự án Khai phá dữ liệu lớn nhằm dự đoán **6 nhóm vũ khí trong PUBG** (`AR`, `SMG`, `Shotgun`, `Other`, `DMR`, `SR`) dựa trên ngữ cảnh không gian - thời gian và lịch sử hạ gục nội trận, xử lý trên toàn bộ hơn **65 triệu sự kiện giao tranh** từ 5 tệp CSV thô.

---

## 1. Tài liệu Dự án

Tất cả báo cáo chi tiết và tài liệu thuyết trình được lưu trong thư mục `docs/`:

* [Báo cáo môn học hoàn chỉnh (PDF)](docs/Bao_cao_PUBG.pdf)
* [Slide thuyết trình (PowerPoint .pptx)](docs/Thuyet_trinh_PUBG.pptx) & [Slide PDF](docs/Thuyet_trinh_PUBG.pdf)

---

## 2. Hướng dẫn Chạy & Tái lập Kết quả (Reproducibility)

### Cách 1: Chạy trực tiếp trên Kaggle (Khuyên dùng - Nhanh nhất)

Toàn bộ quy trình thực nghiệm đã được đóng gói khép kín trong 1 notebook duy nhất trên Kaggle:

**Kaggle Notebook:** [https://www.kaggle.com/code/kieuthithutrang/pubg-weapon-groups-reproduction](https://www.kaggle.com/code/kieuthithutrang/pubg-weapon-groups-reproduction)

**Các bước thực hiện:**

1. Mở liên kết notebook trên Kaggle và nhấn **Copy & Edit**.
2. Tại bảng cấu hình bên phải:
   * **Data:** Gắn bộ dữ liệu [PUBG Match Deaths and Statistics](https://www.kaggle.com/datasets/skihikingkevin/pubg-match-deaths).
   * **Accelerator:** Chọn **GPU T4** hoặc **P100**.
   * **Internet:** Bật **ON** (để tải thư viện TabM, TabICLv2 và clone code).
3. Nhấn **Run All**. Notebook sẽ tự động tiền xử lý, huấn luyện cả 2 quy mô và xuất ra toàn bộ bảng biểu kết quả trong tệp nén `pubg_results_*.zip`.

---

### Cách 2: Chạy trên máy cục bộ (Local Environment)

Yêu cầu môi trường Python 3.11 trở lên (khuyến nghị chạy trên Linux hoặc WSL):

```bash
# 1. Clone repository
git clone https://github.com/kttt294/PUBG-Weapon-Groups.git
cd PUBG-Weapon-Groups

# 2. Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt

# 3. Chạy pipeline thực nghiệm toàn diện (cả 5 CSV)
python -m pubg_weapon.study \
    --source /path/to/deaths/kill_match_stats_final_{0,1,2,3,4}.csv \
    --output artifacts/study
```

---

## 3. Bảng Kết quả Nổi bật

### Bảng 1: Kết quả trên Toàn bộ Dữ liệu Lớn (Tập Test: 6.972.742 dòng)

*Áp dụng kỹ nghệ Out-of-Core trên máy chủ Kaggle RAM 16GB, đỉnh RAM chỉ tốn **4,34 – 4,49 GiB**.*

| Phương pháp                          | Macro-F1 (Độ đo chính) | Khoảng tin cậy 95% CI (Bootstrap) | Accuracy | Ghi chú kỹ thuật                                    |
| :-------------------------------------- | :------------------------: | :---------------------------------: | :------: | :----------------------------------------------------- |
| **Majority Baseline**             |           0,1149           |          [0,1148; 0,1150]          |  52,60%  | Mốc đoán mò lớp đa số (AR)                      |
| **Gaussian Naive Bayes**          |           0,3192           |          [0,3187; 0,3196]          |  38,96%  | Học online theo lô `partial_fit()`                 |
| **Logistic SGD**                  |           0,3954           |          [0,3949; 0,3958]          |  48,39%  | SGDClassifier, L2 penalty ($\alpha = 10^{-4}$)       |
| **XGBoost Hist (Có trọng số)** |      **0,4040**      |     **[0,4036; 0,4045]**     |  45,34%  | **ExtMemQuantileDMatrix 64 bins, max_depth = 6** |

---

### Bảng 2: Kết quả Benchmark Đối chứng SOTA (Tập Test: 9.780 dòng)

*So sánh công bằng trên cùng dữ liệu lấy mẫu nguyên trận, giữ nguyên phân phối tự nhiên giữa các mô hình kinh điển và Deep Learning dữ liệu bảng mới nhất.*

| Phương pháp                 |     Macro-F1     |  Khoảng tin cậy 95% CI  |     Accuracy     | Phân loại kiến trúc                                         |
| :----------------------------- | :--------------: | :------------------------: | :--------------: | :-------------------------------------------------------------- |
| **TabM**                 | **0,4140** | **[0,3984; 0,4281]** |      59,67%      | **Mạng MLP với BatchEnsemble (ICLR 2025) - Vô địch** |
| **TabICLv2**             |      0,4094      |      [0,3938; 0,4235]      | **60,09%** | Tabular Foundation Model In-Context Learning (ICML 2026)        |
| **Random Forest**        |      0,4020      |      [0,3869; 0,4163]      |      59,99%      | 300 cây, max_depth = 18, min_samples_leaf = 5                  |
| **XGBoost Classifier**   |      0,4017      |      [0,3859; 0,4164]      |      59,68%      | Gradient Boosted Trees (max_depth = 6)                          |
| **HistGradientBoosting** |      0,3980      |      [0,3825; 0,4129]      |      58,58%      | GBDT dựa trên Histogram của Scikit-learn                     |
| **Logistic Regression**  |      0,3834      |      [0,3674; 0,3980]      |      58,80%      | Hồi quy Logistic chuẩn tắc                                   |
| **Gaussian Naive Bayes** |      0,3259      |      [0,3138; 0,3381]      |      39,14%      | Xác suất Bayes độc lập                                     |
| **Majority Baseline**    |      0,1146      |      [0,1126; 0,1167]      |      52,41%      | Mốc đoán lớp đa số                                        |
