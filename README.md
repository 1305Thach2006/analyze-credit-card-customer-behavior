# MÔ HÌNH PHÂN TÍCH HÀNH VI TIÊU DÙNG THẺ TÍN DỤNG VÀ ĐỀ XUẤT CHIẾN LƯỢC ƯU ĐÃI CÁ NHÂN

## 1. Giới thiệu tổng quan đề tài
Dự án tập trung giải quyết bài toán cốt lõi trong lĩnh vực ngân hàng bán lẻ và thanh toán số: **Phân tích hành vi tiêu dùng của 50.000 chủ thẻ tín dụng, phân nhóm khách hàng bằng học máy (Machine Learning), kiểm soát rủi ro tài chính và tạo nền tảng số liệu định lượng cho việc đề xuất chiến lược ưu đãi cá nhân hóa**.

Hệ thống được phát triển theo các chuẩn mực kỹ thuật:
- **Môi trường vận hành:** Hoạt động hoàn toàn trên **VS Code** bằng các tệp `.py` thuần túy (không sử dụng Jupyter Notebook, không xây dựng web app).
- **Thuật toán sử dụng:** Kế thừa các giải thuật Machine Learning kinh điển: **K-Means**, **PCA 2D**, **k-NN**, **Linear Regression**, **Decision Tree** và **Random Forest** (loại bỏ hoàn toàn Deep Learning).
- **Chuẩn hóa mã nguồn:** 100% các dòng code đều có chú thích kỹ thuật trực tiếp (`# ...`), tuân thủ nghiêm ngặt mô hình kiến trúc phân tầng dạng mô-đun (Modular Clean Architecture).
- **Phân định trách nhiệm rõ ràng:** Code chỉ xuất số liệu thống kê khách quan và lưu trực tiếp vào các tệp CSV; các chiến lược đề xuất ưu đãi kinh doanh cụ thể được trình bày độc lập trên Báo cáo khoa học 5 trang và Slide thuyết trình.

---

## 2. Cấu trúc kiến trúc phần mềm (Software Architecture)

```
D:\PTDL\
│
├── README.md                    # Tài liệu hướng dẫn kiến trúc và vận hành hệ thống (tại thư mục gốc)
│
├── dataset/                     # TẬNG DỮ LIỆU THỰC NGHIỆM
│   └── synthetic_credit_card_customer_behavior_dataset.csv # 50.000 dòng, 30 thuộc tính giao dịch thẻ
│
└── source_code/                 # TẦNG MÃ NGUỒN VÀ KẾT QUẢ ĐẦU RA
    ├── config/                  # TẦNG CẤU HÌNH HỆ THỐNG
    │   ├── __init__.py
    │   └── settings.py          # Tham số cấu hình, đường dẫn tệp, seed (42), số cụm (4), trọng số thưởng & ngưỡng rủi ro
    │
    ├── data_pipeline/           # TẦNG XỬ LÝ VÀ CHUẨN HÓA DỮ LIỆU
    │   ├── __init__.py
    │   ├── loader.py            # Nạp dữ liệu CSV (50.000 dòng, 30 cột), kiểm tra toàn vẹn và giá trị khuyết
    │   ├── feature_engineering.py # Tính 7 tỷ trọng chi tiêu danh mục (Pct_*), tỷ trọng tùy ý vs thiết yếu, cờ rủi ro (Risk_Flag)
    │   └── preprocessor.py      # Chuẩn hóa Z-score các đặc trưng hành vi bằng StandardScaler
    │
    ├── eda/                     # TẦNG PHÂN TÍCH THỐNG KÊ VÀ TRỰC QUAN HÓA
    │   ├── __init__.py
    │   └── visualizer.py        # Vẽ và xuất đồ thị: Heatmap tương quan, Boxplot 7 danh mục, PCA 2D và Model Benchmark
    │
    ├── models/                  # TẦNG THUẬT TOÁN MACHINE LEARNING
    │   ├── __init__.py
    │   ├── clustering.py        # K-Means (k=4), PCA 2D, kiểm định Silhouette Score và Davies-Bouldin Index
    │   ├── recommender.py       # CustomerBehaviorScorer: Chấm điểm thưởng động, k-NN tìm láng giềng và xuất báo cáo 50.000 dòng
    │   └── validation_models.py # Đối chuẩn mô hình Machine Learning: Linear Regression vs Decision Tree vs Random Forest
    │
    ├── outputs/                 # TẦNG LƯU TRỮ KẾT QUẢ ĐẦU RA
    │   ├── figures/             # 4 biểu đồ trực quan hóa khoa học độ phân giải cao (.png 300 DPI)
    │   │   ├── eda_correlation_heatmap.png        # Ma trận hệ số tương quan Pearson
    │   │   ├── category_spending_boxplot.png      # Phân phối chi tiêu hàng tháng theo 7 danh mục
    │   │   ├── pca_customer_clusters.png          # Trực quan hóa 4 cụm trên không gian PCA 2D
    │   │   └── model_benchmark_comparison.png     # Biểu đồ cột so sánh hệ số R2 giữa các mô hình
    │   └── reports/             # 3 tệp báo cáo số liệu phân tích định lượng (.csv)
    │       ├── customer_level_analysis_report.csv # Báo cáo chi tiết toàn diện 50.000 khách hàng (ID, Cụm, Điểm thưởng, Trạng thái rủi ro)
    │       ├── cluster_personas_summary.csv       # Bảng tổng hợp giá trị trung bình định lượng của 4 cụm Persona
    │       └── model_benchmark_metrics.csv        # Bảng chỉ số đối chuẩn hiệu năng R2, RMSE, MAE của 3 mô hình
    │
    └── main.py                  # Tệp điều phối thực thi toàn bộ quy trình phân tích tự động (Pipeline Orchestrator)
```

---

## 3. Hướng dẫn cài đặt và vận hành trên VS Code

### Bước 1: Mở dự án trong VS Code
Khởi động VS Code và mở thư mục mã nguồn: `D:\PTDL\source_code`.

### Bước 2: Cài đặt các thư viện phụ thuộc
Mở Terminal tích hợp trong VS Code (PowerShell) và cài đặt các thư viện cần thiết:
```powershell
pip install scikit-learn matplotlib seaborn pandas numpy scipy
```

### Bước 3: Thực thi luồng phân tích tự động
Chạy tệp điều phối chính:
```powershell
python main.py
```
*Toàn bộ quá trình nạp dữ liệu, chế tác đặc trưng, phân cụm K-Means, chiếu PCA 2D, chấm điểm hành vi và đối chuẩn mô hình Machine Learning sẽ được tự động kích hoạt và xuất toàn bộ kết quả ra thư mục `outputs/`.*

---

## 4. Chi tiết các sản phẩm đầu ra (Analytical Outputs)

### A. Biểu đồ trực quan hóa (`outputs/figures/`):
1. **`eda_correlation_heatmap.png`:** Ma trận tương quan nhiệt Pearson giữa các biến tài chính (Thu nhập, Chi tiêu, Hạn mức, Điểm tín dụng, Tỷ lệ trả nợ).
2. **`category_spending_boxplot.png`:** Biểu đồ hộp thể hiện trung vị, tứ phân vị và các điểm ngoại lai (Outliers) chi tiêu trên 7 danh mục.
3. **`pca_customer_clusters.png`:** Biểu đồ phân tán trực quan hóa 4 cụm khách hàng trên không gian 2 thành phần chính PCA (giải thích 31.30% phương sai dữ liệu).
4. **`model_benchmark_comparison.png`:** Biểu đồ cột so sánh hệ số xác định $R^2$ giữa Linear Regression, Decision Tree và Random Forest.

### B. Báo cáo dữ liệu định lượng (`outputs/reports/`):
1. **`customer_level_analysis_report.csv`:** Báo cáo chi tiết của **toàn bộ 50.000 khách hàng** gồm 15 chỉ số định lượng: `Customer_ID`, `Cluster`, `Card_Type`, `Monthly_Spending`, `Dominant_Expense_Category`, `Credit_Limit`, `Credit_Utilization`, `Credit_Score`, `Payment_Ratio`, `Dynamic_Reward_Points`, `Risk_Flag`, `Discretionary_Ratio`, `Essential_Ratio`, `Redemption_Rate`, `Risk_Status`.
2. **`cluster_personas_summary.csv`:** Bảng tổng hợp các giá trị trung bình định lượng của 4 cụm khách hàng (quy mô, chi tiêu trung bình, tỷ lệ dùng hạn mức, điểm tín dụng, tỷ trọng chi tiêu).
3. **`model_benchmark_metrics.csv`:** Bảng đo lường hiệu năng 3 mô hình hồi quy:
   - **Linear Regression:** $R^2 = 0.9285$, $\text{RMSE} = 188.46$, $\text{MAE} = 104.01$
   - **Decision Tree Regressor:** $R^2 = 0.9372$, $\text{RMSE} = 176.51$, $\text{MAE} = 77.66$
   - **Random Forest Regressor:** $R^2 = 0.9534$, $\text{RMSE} = 152.18$, $\text{MAE} = 69.21$ *(Mô hình đạt độ chính xác cao nhất)*.

---

## 5. Phương pháp luận và Cơ sở khoa học

### A. Vì sao lựa chọn phân chia 4 cụm ($k = 4$)?
- **Kiểm định toán học:** Chỉ số **Davies-Bouldin Index (DBI)** đạt giá trị đáy tối ưu **2.1151 tại $k = 4$**; đồng thời tại $k=3$, hệ số **Silhouette Score** bị rơi xuống mức đáy ($0.0999$), chứng minh $k=4$ mô tả cấu trúc dữ liệu tốt nhất.
- **Nghiệp vụ ngân hàng:** Phản ánh chính xác **Ma trận 2 chiều kinh điển ($2 \times 2 = 4$ Persona)** kết hợp giữa Trục cấu trúc chi tiêu (Hưởng thụ vs Thiết yếu) và Trục an toàn tài chính (An toàn vs Cảnh báo nợ xấu).

### B. Nguồn gốc các thuật toán:
- **Silhouette Score & Davies-Bouldin Index:** Kế thừa từ nghiên cứu của **Agrawal et al. (2025, Springer)** và **Paper 4** để giải quyết câu hỏi đo lường hiệu quả phân cụm trong **Slide 5.3 (Trang 20)** của ThS. Hồ Hướng Thiên.
- **Decision Tree Regressor:** Nằm trong chương trình học tại **Slide 1.1 (Trang 19)** và bài tập mẫu môn học.
- **Random Forest Regressor:** Bước mở rộng học máy tập hợp (Ensemble Learning), kế thừa từ nghiên cứu ngân hàng quốc tế của **Rajib Sarkar (2026)** để tối ưu hóa độ chính xác dự báo điểm thưởng.
