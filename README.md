# MÔ HÌNH PHÂN TÍCH HÀNH VI TIÊU DÙNG THẺ TÍN DỤNG VÀ ĐỀ XUẤT CHIẾN LƯỢC ƯU ĐÃI CÁ NHÂN

## 1. Giới thiệu tổng quan đề tài
Dự án tập trung giải quyết bài toán cốt lõi trong lĩnh vực ngân hàng bán lẻ và thanh toán số: **Phân tích hành vi tiêu dùng của 50.000 chủ thẻ tín dụng, phân nhóm khách hàng bằng học máy (Machine Learning), kiểm soát rủi ro tài chính và tạo nền tảng số liệu định lượng cho việc đề xuất chiến lược ưu đãi cá nhân hóa**.

Hệ thống ứng dụng các mô hình và giải thuật học máy:
- **Phân nhóm khách hàng:** K-Means kết hợp giảm chiều dữ liệu PCA (Principal Component Analysis 2D).
- **Học máy nhân quả và tối ưu hóa ưu đãi (Causal AI / Uplift Modeling):** Thuật toán Meta-Learners (T-Learner vs S-Learner trên nền LightGBM) ước lượng hiệu ứng can thiệp gia tăng cá nhân hóa CATE, phân nhóm 4 góc phần tư nhân quả kinh điển (Persuadables, Sure Things, Lost Causes, Sleeping Dogs) và kiểm chứng qua đường cong Qini Curve.
- **Dự báo và đánh giá hành vi điểm thưởng:** Đối chuẩn hiệu năng giữa Linear Regression, Random Forest Regressor, LightGBM Regressor và Stacking Hybrid Ensemble.

---

## 2. Cấu trúc kiến trúc phần mềm (Software Architecture)

```
D:\PTDL\
│
├── README.md                    # Tài liệu hướng dẫn kiến trúc và vận hành hệ thống
│
├── dataset/                     # DỮ LIỆU THỰC NGHIỆM
│   └── synthetic_credit_card_customer_behavior_dataset.csv # 50.000 dòng, 30 thuộc tính giao dịch thẻ
│
├── references/                  # TÀI LIỆU NGHIÊN CỨU THAM KHẢO
│   ├── s44196-025-00899-0.pdf
│   ├── Rajib+Sarkar_ajisv7i126459495.pdf
│   ├── Machine+learning.pdf
│   ├── applsci-15-01356-v3.pdf
│   └── (2025)_Enhancing+Credit+and+Charge+Card+Risk+Assessment+Through+Generative+AI+and+Big+Data+Anal[1].pdf
│
└── source_code/                 # MÃ NGUỒN VÀ KẾT QUẢ ĐẦU RA
    ├── config/                  # Cấu hình hệ thống (tham số, ngưỡng rủi ro, trọng số thưởng)
    │   ├── __init__.py
    │   └── settings.py
    │
    ├── data_pipeline/           # Xử lý, nạp dữ liệu và chế tác đặc trưng
    │   ├── __init__.py
    │   ├── loader.py            # Nạp và kiểm tra dữ liệu CSV
    │   ├── feature_engineering.py # Tính tỷ trọng chi tiêu (Pct_*), chi tiêu tùy ý vs thiết yếu, cờ rủi ro
    │   └── preprocessor.py      # Chuẩn hóa Z-score với StandardScaler
    │
    ├── eda/                     # Phân tích thống kê mô tả và trực quan hóa dữ liệu
    │   ├── __init__.py
    │   └── visualizer.py        # Xuất biểu đồ Heatmap, Boxplot, PCA 2D, Qini Curve và Model Benchmark
    │
    ├── models/                  # Thuật toán Machine Learning
    │   ├── __init__.py
    │   ├── clustering.py        # Phân cụm K-Means, giảm chiều PCA 2D, kiểm định Silhouette & Davies-Bouldin
    │   ├── uplift_engine.py     # Học máy nhân quả Uplift (T-Learner, S-Learner, CATE, 4 góc phần tư Uplift)
    │   ├── recommender.py       # Chấm điểm thưởng động, tích hợp phân khúc Uplift và xuất báo cáo
    │   └── validation_models.py # Đối chuẩn mô hình: Linear Regression vs Random Forest vs LightGBM vs Stacking
    │
    ├── outputs/                 # Kết quả đầu ra của mô hình
    │   ├── figures/             # 6 biểu đồ trực quan hóa khoa học (.png 300 DPI, chú thích tiếng Việt)
    │   │   ├── eda_correlation_heatmap.png        # Ma trận tương quan Pearson
    │   │   ├── category_spending_boxplot.png      # Phân phối chi tiêu theo 7 danh mục
    │   │   ├── kmeans_k_optimization_curve.png    # Đánh giá tối ưu hóa số cụm k (2 → 6) qua Elbow, Silhouette, DBI
    │   │   ├── pca_customer_clusters.png          # Phân cụm khách hàng trên không gian PCA 2D
    │   │   ├── uplift_qini_curve.png              # Đường cong Qini Curve đánh giá T-Learner vs S-Learner
    │   │   └── model_benchmark_comparison.png     # Biểu đồ cột so sánh hệ số R2 giữa 4 mô hình Machine Learning & Hybrid
    │   └── reports/             # 5 báo cáo phân tích định lượng (.csv)
    │       ├── customer_level_analysis_report.csv # Báo cáo chi tiết toàn diện 50.000 khách hàng (kèm Uplift)
    │       ├── cluster_personas_summary.csv       # Tổng hợp giá trị trung bình định lượng của các cụm
    │       ├── k_selection_metrics.csv            # Chỉ số đánh giá WCSS, Silhouette, DBI theo k = 2 → 6
    │       ├── uplift_segments_summary.csv        # Báo cáo định lượng 4 nhóm nhân quả Uplift Modeling
    │       └── model_benchmark_metrics.csv        # Chỉ số đối chuẩn hiệu năng R2, RMSE, MAE của 4 mô hình
    │
    └── main.py                  # Tệp điều phối thực thi toàn bộ quy trình phân tích tự động
```

---

## 3. Hướng dẫn cài đặt và vận hành

### Bước 1: Khởi động môi trường
Mở thư mục mã nguồn: `D:\PTDL\source_code`.

### Bước 2: Cài đặt các thư viện phụ thuộc
Chạy lệnh cài đặt các gói thư viện cần thiết trong Terminal (PowerShell):
```powershell
pip install scikit-learn lightgbm xgboost matplotlib seaborn pandas numpy scipy scikit-uplift
```

### Bước 3: Thực thi luồng phân tích tự động
Chạy tệp điều phối chính:
```powershell
python main.py
```
*Toàn bộ quá trình nạp dữ liệu, chế tác đặc trưng, phân cụm K-Means, chiếu PCA 2D, mô hình hóa học máy nhân quả Uplift Modeling, chấm điểm hành vi và đối chuẩn mô hình Machine Learning sẽ được tự động kích hoạt và xuất toàn bộ kết quả ra thư mục `outputs/`.*

---

## 4. Chi tiết các sản phẩm đầu ra (Analytical Outputs)

### A. Biểu đồ trực quan hóa (`outputs/figures/`):
1. **`eda_correlation_heatmap.png`:** Ma trận tương quan nhiệt Pearson giữa các biến tài chính (Thu nhập, Chi tiêu, Hạn mức, Điểm tín dụng, Tỷ lệ trả nợ).
2. **`category_spending_boxplot.png`:** Biểu đồ hộp thể hiện trung vị, tứ phân vị và các điểm ngoại lai (Outliers) chi tiêu trên 7 danh mục.
3. **`kmeans_k_optimization_curve.png`:** Biểu đồ 3 đồ thị con đánh giá tối ưu hóa số lượng phân cụm $k \in [2, 6]$ theo phương pháp Elbow (WCSS), Silhouette Score và Davies-Bouldin Index (DBI), đánh dấu điểm chọn tối ưu $k = 4$.
4. **`pca_customer_clusters.png`:** Biểu đồ phân tán trực quan hóa các cụm khách hàng trên không gian 2 thành phần chính PCA.
5. **`uplift_qini_curve.png`:** Đường cong Qini biểu diễn số lượng khách hàng chuyển đổi gia tăng lũy tích khi tiếp cận theo thứ hạng điểm Uplift Score, so sánh hiệu năng T-Learner, S-Learner với đường cơ sở lựa chọn ngẫu nhiên.
6. **`model_benchmark_comparison.png`:** Biểu đồ cột so sánh đối chuẩn hệ số xác định $R^2$ giữa 4 mô hình hồi quy (gồm mô hình cơ sở, 2 mô hình học máy tập hợp và mô hình lai Stacking Hybrid).

### B. Báo cáo dữ liệu định lượng (`outputs/reports/`):
1. **`customer_level_analysis_report.csv`:** Báo cáo chi tiết toàn diện cho toàn bộ khách hàng gồm 17 chỉ số định lượng: `Customer_ID`, `Cluster`, `Card_Type`, `Monthly_Spending`, `Dominant_Expense_Category`, `Credit_Limit`, `Credit_Utilization`, `Credit_Score`, `Payment_Ratio`, `Dynamic_Reward_Points`, `Risk_Flag`, `Discretionary_Ratio`, `Essential_Ratio`, `Redemption_Rate`, `Risk_Status`, `Uplift_Score`, `Uplift_Segment`.
2. **`cluster_personas_summary.csv`:** Bảng tổng hợp các giá trị trung bình định lượng của các cụm khách hàng (quy mô cụm, chi tiêu trung bình, tỷ lệ sử dụng hạn mức, điểm tín dụng, tỷ trọng chi tiêu).
3. **`k_selection_metrics.csv`:** Bảng thực nghiệm đo lường 3 chỉ số Inertia (WCSS), Silhouette Score và Davies-Bouldin Index theo từng giá trị $k$ từ 2 đến 6.
4. **`uplift_segments_summary.csv`:** Bảng tổng hợp định lượng 4 nhóm nhân quả Uplift: quy mô khách hàng, tỷ lệ phân bổ (%), điểm Uplift trung bình, chi tiêu hàng tháng, hạn mức, điểm tín dụng, tỷ lệ dùng hạn mức và tỷ lệ thanh toán dư nợ.
5. **`model_benchmark_metrics.csv`:** Bảng đo lường hiệu năng đối chuẩn giữa 4 mô hình (`Linear Regression`, `Random Forest Regressor`, `LightGBM Regressor`, `Stacking Hybrid Ensemble`) theo các thước đo $R^2$, RMSE, MAE.

---

## 5. Phương pháp luận và Cơ sở khoa học của các mô hình

Hệ thống xây dựng dựa trên sự kết hợp giữa các mô hình kế thừa từ các công trình nghiên cứu khoa học quốc tế trong thư mục `references/` và các giải pháp học máy tiên tiến phù hợp với bài toán thẻ tín dụng:

### A. Các mô hình kế thừa trực tiếp từ bài nghiên cứu (trong `references/`):
1. **K-Means Clustering:**
   - *Kế thừa từ bài báo:* **"Analyzing and Rewarding Credit Card Spending Habits in India: a Machine Learning Approach"** (Renuka Agrawal, Aryan Khanna, Safa Hamdare, 2025, *International Journal of Computational Intelligence Systems*, Springer).
   - *Ứng dụng:* Phân khúc cơ sở khách hàng thành các nhóm hành vi tiêu dùng riêng biệt dựa trên tỷ trọng chi tiêu danh mục.
2. **Silhouette Score & Davies-Bouldin Index (DBI):**
   - *Kế thừa từ bài báo:* **"Analyzing and Rewarding Credit Card Spending Habits in India: a Machine Learning Approach"** (Agrawal et al., 2025) và **"Machine Learning in Financial Risk and Behavior Analysis: Predictive Insights on Bankruptcy, Fraud, and Consumer Trends in the USA"** (Maksuda Begum, 2025).
   - *Ứng dụng:* Bộ đôi thước đo toán học định lượng để đánh giá chất lượng phân cụm và xác định số cụm tối ưu ($k = 4$).
3. **Random Forest Regressor (Ensemble Bagging):**
   - *Kế thừa từ bài báo:* **"Analyzing and Rewarding Credit Card Spending Habits in India: a Machine Learning Approach"** (Agrawal et al., 2025) và **"A Systematic Review of AI-Driven Credit Risk Assessment Models in Commercial Banking (2018-2026)"** (Rajib Sarkar, 2026).
   - *Ứng dụng:* Kỹ thuật tập hợp nhiều cây quyết định độc lập song song nhằm triệt tiêu phương sai (Variance Reduction), hạn chế rủi ro Overfitting trên dữ liệu giao dịch thẻ có độ biến động cao.
4. **LightGBM Regressor (Gradient Boosting):**
   - *Kế thừa từ bài báo:* **"A Systematic Review of AI-Driven Credit Risk Assessment Models in Commercial Banking (2018-2026)"** (Rajib Sarkar, 2026).
   - *Ứng dụng:* Phương pháp Gradient Boosted Decision Trees (GBDT) hiện đại, tối ưu hóa phần dư sai số theo chuỗi tuần tự với tốc độ xử lý nhanh và độ chính xác cao trên dữ liệu dạng bảng quy mô lớn của ngành ngân hàng.
5. **PCA (Principal Component Analysis 2D):**
   - *Kế thừa từ bài báo:* **"Machine Learning in Financial Risk and Behavior Analysis: Predictive Insights on Bankruptcy, Fraud, and Consumer Trends in the USA"** (Maksuda Begum, 2025).
   - *Ứng dụng:* Giảm chiều không gian đặc trưng hành vi xuống 2 thành phần chính để trực quan hóa sự phân tách giữa các cụm khách hàng.

### B. Các mô hình và giải thuật học máy chuyên sâu được tích hợp:
1. **Uplift Modeling (Causal Machine Learning / Meta-Learners: T-Learner vs S-Learner):**
   - *Cơ sở khoa học:* Trong chiến lược ngân hàng số hiện đại, mô hình dự báo xác suất mua hàng truyền thống (Response Modeling) chỉ cho biết ai có khả năng chuyển đổi cao, dẫn đến lãng phí ngân sách tiếp thị cho những khách hàng vốn dĩ tự mua (Sure Things). Học máy nhân quả Uplift Modeling giải quyết triệt để vấn đề này bằng cách ước lượng hiệu ứng can thiệp biên cá nhân hóa CATE (Conditional Average Treatment Effect):
     $$\tau(X) = \mathbb{E}[Y | X, W=1] - \mathbb{E}[Y | X, W=0]$$
   - *Ứng dụng:* Hệ thống triển khai hai Meta-Learners kết hợp bộ phân loại LightGBM:
     - **T-Learner (Two Models):** Huấn luyện 2 mô hình độc lập trên tập có ưu đãi ($W=1$) và tập đối chứng ($W=0$), triệt tiêu thiên lệch tác động.
     - **S-Learner (Solo Model):** Huấn luyện 1 mô hình tổng thể chứa biến can thiệp làm thuộc tính đầu vào.
   - *Hiệu quả thực nghiệm:* Đạt Qini AUC Score 0.1215 và Uplift@30% đạt 0.3370 (khi tập trung ngân sách vào 30% khách hàng có điểm Uplift cao nhất, tỷ lệ chuyển đổi gia tăng biên đạt +33.70% so với phân bổ ngẫu nhiên). Mô hình phân bổ chính xác 50.000 khách hàng vào 4 góc phần tư: **Persuadables** (40.0540% - đối tượng vàng nhận ưu đãi), **Sure Things** (34.4280% - tự mua, cắt giảm voucher tiết kiệm chi phí), **Lost Causes** (20.7320% - rủi ro cao, áp dụng van an toàn kiểm soát nợ), **Sleeping Dogs** (4.7860% - nhạy cảm với quảng cáo, ngừng làm phiền).
2. **Linear Regression (Hồi quy tuyến tính):**
   - *Lý do lựa chọn:* Đóng vai trò là mô hình cơ sở (Baseline Model) trong bài toán hồi quy điểm thưởng hành vi. Phù hợp để kiểm chứng mối tương quan tuyến tính chuẩn mực giữa các biến tài chính (Hạn mức, Chi tiêu, Thu nhập) trước khi đối chuẩn với các mô hình phi tuyến tính phức tạp hơn.
3. **Stacking Hybrid Ensemble (Mô hình lai học máy tập hợp):**
   - *Lý do lựa chọn:* Kế thừa kiến trúc Ensemble Stacking tiên tiến từ các nghiên cứu quốc tế trong ngành ngân hàng thương mại (Rajib Sarkar, 2026; Begum, 2025). Mô hình kết hợp sức mạnh của hai trường phái học máy tập hợp bổ trợ cho nhau: Random Forest (Bagging - triệt tiêu phương sai, kháng nhiễu) và LightGBM (Boosting - tối ưu hóa chuỗi phần dư sai số), thông qua mô hình trọng tài Linear Regression (Meta-Learner). Mô hình lai đạt hiệu năng tổng thể cao nhất toàn bộ hệ thống ($R^2 = 0.9314$, RMSE = 150.0302).
