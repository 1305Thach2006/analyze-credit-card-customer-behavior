# Import thư viện warnings để ẩn các cảnh báo phiên bản tương lai của scikit-learn
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Import thư viện os để thao tác với đường dẫn và tệp tin
import os
# Import thư viện matplotlib để vẽ đồ thị khoa học
import matplotlib.pyplot as plt
# Import thư viện seaborn để trực quan hóa thống kê
import seaborn as sns
# Import thư viện pandas để thao tác dữ liệu
import pandas as pd
# Import thư viện numpy để tính toán ma trận
import numpy as np

# Thiết lập phông chữ hỗ trợ đầy đủ tiếng Việt có dấu trên mọi hệ điều hành
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'DejaVu Sans']
# Thiết lập hiển thị dấu trừ chính xác trong biểu đồ
plt.rcParams['axes.unicode_minus'] = False
# Cài đặt phong cách mặc định cho các biểu đồ trực quan hóa
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Hàm lưu biểu đồ an toàn chống lỗi PermissionError khi người dùng đang mở xem ảnh
def save_plot_safely(output_path: str, dpi: int = 300) -> str:
    # Khối lệnh thử lưu tệp ảnh gốc
    try:
        # Lưu hình ảnh biểu đồ ra tập tin với độ phân giải cao
        plt.savefig(output_path, dpi=dpi)
        # Đóng khung vẽ để giải phóng tài nguyên bộ nhớ
        plt.close()
        # In thông báo đã xuất tập tin ảnh thành công
        print(f"[EDA] Đã lưu biểu đồ tại: {output_path}")
        # Trả về đường dẫn gốc
        return output_path
    # Bắt lỗi PermissionError khi tệp ảnh đang mở trong trình xem ảnh
    except PermissionError:
        # Tách tên tập tin và đuôi mở rộng
        base, ext = os.path.splitext(output_path)
        # Tạo đường dẫn thay thế với hậu tố _updated
        alt_path = f"{base}_updated{ext}"
        # Lưu hình ảnh sang đường dẫn dự phòng mới
        plt.savefig(alt_path, dpi=dpi)
        # Đóng khung vẽ để giải phóng bộ nhớ
        plt.close()
        # In cảnh báo tập tin đang mở và thông báo đường dẫn mới
        print(f"[CẢNH BÁO] Tập tin {os.path.basename(output_path)} đang mở trong trình xem ảnh. Đã lưu vào: {alt_path}")
        # Trả về đường dẫn mới
        return alt_path

# Hàm vẽ và xuất biểu đồ ma trận nhiệt hệ số tương quan giữ nguyên 7 tên thuộc tính gốc
def plot_correlation_heatmap(df: pd.DataFrame, columns: list, output_path: str) -> None:
    # Tạo khung hình vẽ với kích thước 10x8 inch
    plt.figure(figsize=(10, 8))
    # Tính toán ma trận hệ số tương quan tuyến tính Pearson giữ nguyên tên thuộc tính gốc
    corr_matrix = df[columns].corr()
    # Vẽ biểu đồ nhiệt với bảng màu coolwarm và hiển thị hệ số làm tròn 2 chữ số
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5, cbar=True)
    # Đặt tiêu đề tiếng Việt có dấu cho biểu đồ tương quan
    plt.title('Ma trận hệ số tương quan giữa các đặc trưng chi tiêu và tài chính', fontsize=12, fontweight='bold')
    # Căn chỉnh bố cục khung vẽ
    plt.tight_layout()
    # Gọi hàm lưu ảnh an toàn chống khóa tệp
    save_plot_safely(output_path, dpi=300)

# Hàm vẽ biểu đồ hộp thể hiện phân phối chi tiêu theo 7 danh mục giữ nguyên tên thuộc tính
def plot_category_spending_boxplot(df: pd.DataFrame, spending_cols: list, output_path: str) -> None:
    # Tạo khung hình vẽ với kích thước 12x6 inch
    plt.figure(figsize=(12, 6))
    # Chuyển dữ liệu sang dạng dài để vẽ biểu đồ hộp cho từng danh mục
    melted_df = df[spending_cols].melt(var_name='Danh_Muc', value_name='Chi_Tieu_Thang')
    # Rút gọn tên danh mục bỏ hậu tố _Spending để trục hoành gọn gàng
    melted_df['Danh_Muc'] = melted_df['Danh_Muc'].str.replace('_Spending', '')
    # Vẽ biểu đồ boxplot thể hiện trung vị và khoảng tứ phân vị
    sns.boxplot(x='Danh_Muc', y='Chi_Tieu_Thang', data=melted_df, hue='Danh_Muc', palette='Set2', legend=False)
    # Đặt tiêu đề tiếng Việt có dấu cho biểu đồ
    plt.title('Phân phối chi tiêu hàng tháng theo 7 danh mục tiêu dùng', fontsize=12, fontweight='bold')
    # Đặt nhãn trục hoành bằng tiếng Việt
    plt.xlabel('Danh mục chi tiêu', fontsize=11, fontweight='bold')
    # Đặt nhãn trục tung bằng tiếng Việt
    plt.ylabel('Số tiền chi tiêu hàng tháng (VNĐ)', fontsize=11, fontweight='bold')
    # Căn chỉnh bố cục khung vẽ
    plt.tight_layout()
    # Gọi hàm lưu ảnh an toàn chống khóa tệp
    save_plot_safely(output_path, dpi=300)

# Hàm vẽ biểu đồ phân cụm khách hàng trên không gian giảm chiều PCA 2D
def plot_pca_clusters(pca_2d: np.ndarray, labels: np.ndarray, output_path: str) -> None:
    # Tạo khung hình vẽ với kích thước 9x7 inch
    plt.figure(figsize=(9, 7))
    # Định nghĩa danh sách 4 màu sắc tương ứng với 4 cụm
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    # Định nghĩa tên nhãn cụm đồng bộ với báo cáo và chân dung 4 nhóm
    cluster_names = [
        # Nhãn đại diện cho cụm 0
        'Cụm 0: Family & Essentials',
        # Nhãn đại diện cho cụm 1
        'Cụm 1: VIP Lifestyle Spenders',
        # Nhãn đại diện cho cụm 2
        'Cụm 2: Commuters & Everyday',
        # Nhãn đại diện cho cụm 3
        'Cụm 3: Utility Bill Spenders'
    # Kết thúc danh sách tên cụm
    ]
    # Lặp qua từng cụm để vẽ các điểm dữ liệu
    for cluster_id in range(len(colors)):
        # Lọc các điểm dữ liệu thuộc cụm hiện tại
        cluster_mask = (labels == cluster_id)
        # Vẽ biểu đồ phân tán cho các điểm thuộc cụm với đầy đủ tham số trực quan
        plt.scatter(pca_2d[cluster_mask, 0], pca_2d[cluster_mask, 1], c=colors[cluster_id], label=cluster_names[cluster_id], alpha=0.6, edgecolors='w', s=35)
    # Đặt tiêu đề tiếng Việt có dấu cho biểu đồ phân cụm
    plt.title('Không gian phân cụm khách hàng thẻ tín dụng trên PCA 2D', fontsize=12, fontweight='bold')
    # Đặt nhãn cho trục thành phần chính thứ nhất
    plt.xlabel('Thành phần chính 1 (PCA 1: Hưởng thụ vs Thiết yếu)', fontsize=11, fontweight='bold')
    # Đặt nhãn cho trục thành phần chính thứ hai
    plt.ylabel('Thành phần chính 2 (PCA 2: Tạp hóa vs Hóa đơn / Xăng dầu)', fontsize=11, fontweight='bold')
    # Hiển thị chú thích các cụm bằng tiếng Việt
    plt.legend(title='Phân khúc khách hàng', loc='best', frameon=True)
    # Căn chỉnh bố cục khung vẽ
    plt.tight_layout()
    # Gọi hàm lưu ảnh an toàn chống khóa tệp
    save_plot_safely(output_path, dpi=300)

# Hàm vẽ biểu đồ tối ưu hóa số cụm k từ 2 đến 6 gồm 3 chỉ số Elbow, Silhouette và DBI
def plot_kmeans_k_optimization(k_results_df: pd.DataFrame, output_path: str) -> None:
    # Tạo khung hình vẽ gồm 3 đồ thị con nằm ngang với kích thước 15x4.5 inch
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    # Lấy danh sách giá trị k
    k_vals = k_results_df['k'].values
    # Đồ thị con 1: Phương pháp Elbow thể hiện quán tính nội cụm WCSS
    axes[0].plot(k_vals, k_results_df['Inertia_WCSS'], marker='o', color='#1565c0', linewidth=2, markersize=7)
    # Vẽ đường gióng nét đứt màu đỏ tại k = 4 để đánh dấu điểm tối ưu
    axes[0].axvline(x=4, color='#d32f2f', linestyle='--', linewidth=1.5, label='Điểm chọn k = 4')
    # Đặt tiêu đề tiếng Việt có dấu cho đồ thị Elbow
    axes[0].set_title('Phương pháp Elbow (Inertia / WCSS)', fontsize=11, fontweight='bold')
    # Đặt nhãn trục hoành
    axes[0].set_xlabel('Số lượng cụm (k)', fontsize=10, fontweight='bold')
    # Đặt nhãn trục tung
    axes[0].set_ylabel('Tổng bình phương khoảng cách (WCSS)', fontsize=10)
    # Bật lưới tọa độ cho đồ thị con 1
    axes[0].grid(True, linestyle=':', alpha=0.6)
    # Hiển thị chú thích điểm chọn k = 4
    axes[0].legend(loc='upper right')
    # Đồ thị con 2: Hệ số phân tách Silhouette Score
    axes[1].plot(k_vals, k_results_df['Silhouette_Score'], marker='s', color='#2e7d32', linewidth=2, markersize=7)
    # Vẽ đường gióng nét đứt màu đỏ tại k = 4
    axes[1].axvline(x=4, color='#d32f2f', linestyle='--', linewidth=1.5, label='Điểm chọn k = 4')
    # Đặt tiêu đề tiếng Việt có dấu cho đồ thị Silhouette
    axes[1].set_title('Hệ số phân tách Silhouette Score', fontsize=11, fontweight='bold')
    # Đặt nhãn trục hoành
    axes[1].set_xlabel('Số lượng cụm (k)', fontsize=10, fontweight='bold')
    # Đặt nhãn trục tung
    axes[1].set_ylabel('Silhouette Score (Càng cao càng tốt)', fontsize=10)
    # Bật lưới tọa độ cho đồ thị con 2
    axes[1].grid(True, linestyle=':', alpha=0.6)
    # Hiển thị chú thích
    axes[1].legend(loc='lower right')
    # Đồ thị con 3: Chỉ số đo đặc cụm Davies-Bouldin Index (DBI)
    axes[2].plot(k_vals, k_results_df['Davies_Bouldin_Index'], marker='^', color='#e65100', linewidth=2, markersize=7)
    # Vẽ đường gióng nét đứt màu đỏ tại k = 4
    axes[2].axvline(x=4, color='#d32f2f', linestyle='--', linewidth=1.5, label='Điểm chọn k = 4')
    # Đặt tiêu đề tiếng Việt có dấu cho đồ thị DBI
    axes[2].set_title('Chỉ số Davies-Bouldin Index (DBI)', fontsize=11, fontweight='bold')
    # Đặt nhãn trục hoành
    axes[2].set_xlabel('Số lượng cụm (k)', fontsize=10, fontweight='bold')
    # Đặt nhãn trục tung
    axes[2].set_ylabel('Davies-Bouldin Index (Càng thấp càng tốt)', fontsize=10)
    # Bật lưới tọa độ cho đồ thị con 3
    axes[2].grid(True, linestyle=':', alpha=0.6)
    # Hiển thị chú thích
    axes[2].legend(loc='upper right')
    # Đặt tiêu đề tổng thể cho toàn bộ khung biểu đồ
    fig.suptitle('Đánh giá và tối ưu hóa số lượng phân cụm k (k = 2 → 6) bằng Elbow, Silhouette và DBI', fontsize=13, fontweight='bold')
    # Căn chỉnh bố cục để các đồ thị con không đè lên nhau
    plt.tight_layout()
    # Gọi hàm lưu ảnh an toàn chống khóa tệp
    save_plot_safely(output_path, dpi=300)

# Hàm vẽ biểu đồ cột so sánh độ chính xác R2 giữa các mô hình dự báo và mô hình lai
def plot_model_benchmark(benchmark_df: pd.DataFrame, output_path: str) -> None:
    # Tạo khung hình vẽ với kích thước 10x5.5 inch
    plt.figure(figsize=(10, 5.5))
    # Danh sách màu sắc trực quan cho 4 mô hình
    bar_colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    # Từ điển ánh xạ tên mô hình sang tiếng Việt có dấu để hiển thị trên trục hoành
    model_vn_labels = {
        # Mô hình hồi quy tuyến tính cơ sở
        'Linear Regression': 'Linear Regression\n(Cơ sở)',
        # Mô hình tập hợp Random Forest
        'Random Forest Regressor': 'Random Forest\n(Bagging)',
        # Mô hình tăng cường LightGBM
        'LightGBM Regressor': 'LightGBM\n(Boosting)',
        # Mô hình lai xếp chồng Stacking Hybrid
        'Stacking Hybrid Ensemble': 'Stacking Hybrid\n(Mô hình Lai SOTA)'
    # Kết thúc từ điển ánh xạ
    }
    # Tạo danh sách nhãn trục hoành tiếng Việt
    display_names = [model_vn_labels.get(name, name) for name in benchmark_df['Model']]
    # Vẽ biểu đồ cột thể hiện hệ số xác định R2 của các mô hình
    bars = plt.bar(display_names, benchmark_df['R2_Score'], color=bar_colors[:len(display_names)], width=0.45)
    # Đặt tiêu đề tiếng Việt có dấu cho biểu đồ
    plt.title('Đối chuẩn hiệu năng hệ số xác định R² giữa các mô hình học máy và mô hình lai (Hybrid)', fontsize=12, fontweight='bold')
    # Đặt nhãn cho trục tung bằng tiếng Việt
    plt.ylabel('Hệ số xác định R² (R-squared Score)', fontsize=11, fontweight='bold')
    # Giới hạn trục tung từ 0 đến 1.1 để hiển thị rõ giá trị
    plt.ylim(0, 1.1)
    # Hiển thị giá trị cụ thể trên đầu từng cột
    for bar in bars:
        # Lấy chiều cao của cột
        yval = bar.get_height()
        # In giá trị lên trên đỉnh cột với định dạng đậm nét 4 chữ số thập phân
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    # Căn chỉnh bố cục khung vẽ
    plt.tight_layout()
    # Gọi hàm lưu ảnh an toàn chống khóa tệp
    save_plot_safely(output_path, dpi=300)

# Hàm vẽ biểu đồ đường cong Qini đối chuẩn hiệu năng các mô hình Uplift Modeling
def plot_uplift_qini_curve(y_test: np.ndarray, uplift_t: np.ndarray, uplift_s: np.ndarray, trt_test: np.ndarray, output_path: str) -> None:
    # Import hàm qini_curve từ scikit-uplift
    from sklift.metrics import qini_curve, qini_auc_score
    # Tạo khung hình vẽ với kích thước 10x6 inch
    plt.figure(figsize=(10, 6))

    # Tính toán tọa độ đường cong Qini cho mô hình T-Learner
    x_t, y_t = qini_curve(y_test, uplift_t, trt_test)
    # Tính tọa độ đường cong Qini cho mô hình S-Learner
    x_s, y_s = qini_curve(y_test, uplift_s, trt_test)

    # Tính điểm Qini AUC cho T-Learner
    qini_score_t = qini_auc_score(y_test, uplift_t, trt_test)
    # Tính điểm Qini AUC cho S-Learner
    qini_score_s = qini_auc_score(y_test, uplift_s, trt_test)

    # Vẽ đường cong Qini của T-Learner (màu xanh dương đậm SOTA) với 4 chữ số thập phân
    plt.plot(x_t, y_t, label=f'T-Learner (TwoModels) - Qini Score: {qini_score_t:.4f}', color='#1E40AF', linewidth=2.5)
    # Vẽ đường cong Qini của S-Learner (màu vàng cam) với 4 chữ số thập phân
    plt.plot(x_s, y_s, label=f'S-Learner (SoloModel) - Qini Score: {qini_score_s:.4f}', color='#D97706', linewidth=2.0)

    # Vẽ đường tiếp cận ngẫu nhiên (Random Baseline) từ gốc (0,0) đến điểm cuối
    plt.plot([0, x_t[-1]], [0, y_t[-1]], label='Random Targeting (Cơ sở ngẫu nhiên)', color='#64748B', linestyle='--', linewidth=1.5)

    # Tô màu dải hiệu năng vượt trội giữa T-Learner và Random baseline
    baseline_y = np.linspace(0, y_t[-1], len(x_t))
    plt.fill_between(x_t, y_t, baseline_y, color='#1E40AF', alpha=0.1, label='Hiệu quả gia tăng (Incremental Uplift Gain)')

    # Đặt tiêu đề biểu đồ bằng tiếng Việt có dấu
    plt.title('Đường cong Qini (Qini Curve) đối chuẩn mô hình Uplift Modeling dự báo hiệu ứng ưu đãi', fontsize=12, fontweight='bold')
    # Đặt nhãn trục hoành
    plt.xlabel('Số lượng khách hàng được can thiệp ưu đãi (Targeted Population)', fontsize=11, fontweight='bold')
    # Đặt nhãn trục tung
    plt.ylabel('Chuyển đổi gia tăng tích lũy (Cumulative Incremental Gain)', fontsize=11, fontweight='bold')
    # Bật lưới tọa độ
    plt.grid(True, linestyle=':', alpha=0.6)
    # Hiển thị chú thích ở góc trên bên trái
    plt.legend(loc='upper left', fontsize=10, frameon=True)

    # Căn chỉnh bố cục khung vẽ
    plt.tight_layout()
    # Gọi hàm lưu ảnh an toàn chống khóa tệp
    save_plot_safely(output_path, dpi=300)

