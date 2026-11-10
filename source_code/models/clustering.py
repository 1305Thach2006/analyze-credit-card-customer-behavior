# Import thư viện KMeans từ scikit-learn để phân cụm dữ liệu
from sklearn.cluster import KMeans
# Import thư viện PCA từ scikit-learn để giảm chiều dữ liệu
from sklearn.decomposition import PCA
# Import thư viện tính chỉ số đo lường phân tách Silhouette Score
from sklearn.metrics import silhouette_score
# Import thư viện tính chỉ số đánh giá độ đặc cụm Davies-Bouldin Index
from sklearn.metrics import davies_bouldin_score
# Import thư viện numpy để tính toán ma trận và mảng số học
import numpy as np
# Import thư viện pandas để quản lý bảng kết quả thống kê
import pandas as pd
# Import thư viện typing để định nghĩa kiểu dữ liệu
from typing import Dict, Any, Tuple

# Lớp thực thi quy trình phân cụm bằng thuật toán K-Means và PCA
class CustomerClustering:
    # Hàm khởi tạo lớp CustomerClustering với số cụm và hạt giống ngẫu nhiên
    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        # Lưu số lượng cụm cần phân chia
        self.n_clusters = n_clusters
        # Lưu giá trị hạt giống để cố định kết quả
        self.random_state = random_state
        # Khởi tạo mô hình KMeans sử dụng thuật toán k-means++ khởi tạo tâm cụm
        self.kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', n_init=10, max_iter=300, random_state=self.random_state)
        # Khởi tạo đối tượng PCA giảm xuống 2 chiều để trực quan hóa
        self.pca = PCA(n_components=2, random_state=self.random_state)
        # Biến lưu trữ mảng nhãn cụm của toàn bộ tập dữ liệu
        self.labels_ = None

    # Hàm chạy thực nghiệm quét các giá trị k từ 2 đến 6 để kiểm định và tối ưu hóa số cụm
    def evaluate_k_range(self, X: np.ndarray, k_range=range(2, 7), sample_size: int = 10000) -> pd.DataFrame:
        # In thông báo bắt đầu thực nghiệm quét số cụm k
        print(f"[CLUSTERING] Bắt đầu thực nghiệm quét số cụm k từ {min(k_range)} đến {max(k_range)}...")
        # Danh sách lưu trữ kết quả đánh giá của từng giá trị k
        k_results = []
        # Kiểm tra nếu dữ liệu lớn hơn sample_size thì lấy mẫu để tính toán nhanh chóng
        if len(X) > sample_size:
            # Khởi tạo bộ sinh số ngẫu nhiên với seed cố định
            rng = np.random.RandomState(self.random_state)
            # Chọn các chỉ số mẫu ngẫu nhiên
            sample_idx = rng.choice(len(X), size=sample_size, replace=False)
            # Trích xuất tập ma trận mẫu để đánh giá Silhouette và DBI
            X_eval = X[sample_idx]
        # Trường hợp dữ liệu nhỏ hơn sample_size
        else:
            # Sử dụng toàn bộ ma trận đặc trưng
            X_eval = X
            # Đặt sample_idx là None
            sample_idx = None
        # Lặp qua từng giá trị k trong dải k_range
        for k in k_range:
            # Khởi tạo mô hình KMeans với k hiện tại
            km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=self.random_state)
            # Thực hiện huấn luyện và gán nhãn cụm trên toàn bộ ma trận X
            labels_all = km.fit_predict(X)
            # Lấy giá trị quán tính nội cụm WCSS
            inertia = float(km.inertia_)
            # Trích xuất nhãn cụm tương ứng với tập mẫu đánh giá
            labels_eval = labels_all[sample_idx] if sample_idx is not None else labels_all
            # Tính hệ số phân tách Silhouette Score trên tập mẫu
            sil = float(silhouette_score(X_eval, labels_eval))
            # Tính chỉ số Davies-Bouldin Index trên tập mẫu
            dbi = float(davies_bouldin_score(X_eval, labels_eval))
            # In kết quả chỉ số của k hiện tại ra console làm tròn 4 chữ số thập phân
            print(f"[CLUSTERING] -> k = {k}: WCSS = {inertia:.2f} | Silhouette = {sil:.4f} | DBI = {dbi:.4f}")
            # Lưu các thông số vào danh sách kết quả làm tròn 4 chữ số thập phân
            k_results.append({
                # Giá trị số cụm k đang thử nghiệm
                'k': k,
                # Tổng bình phương khoảng cách nội cụm WCSS (Inertia)
                'Inertia_WCSS': round(inertia, 2),
                # Hệ số tách biệt Silhouette Score
                'Silhouette_Score': round(sil, 4),
                # Chỉ số đánh giá độ đặc Davies-Bouldin Index
                'Davies_Bouldin_Index': round(dbi, 4)
            # Kết thúc khối phần tử danh sách
            })
        # Chuyển đổi danh sách thành DataFrame của pandas
        results_df = pd.DataFrame(k_results)
        # Trả về bảng kết quả thực nghiệm
        return results_df

    # Hàm huấn luyện và dự đoán nhãn cụm cho toàn bộ khách hàng
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        # In thông báo bắt đầu phân cụm K-Means
        print(f"[CLUSTERING] Bắt đầu phân cụm K-Means với k = {self.n_clusters}...")
        # Thực hiện huấn luyện và gán nhãn cụm cho tập dữ liệu
        self.labels_ = self.kmeans.fit_predict(X)
        # In thông báo đã hoàn thành phân cụm
        print("[CLUSTERING] Phân cụm K-Means hoàn tất!")
        # Trả về mảng chứa nhãn cụm của các khách hàng
        return self.labels_

    # Hàm đánh giá chất lượng phân cụm bằng các chỉ số khoa học
    def evaluate(self, X: np.ndarray, sample_size: int = 10000) -> Dict[str, float]:
        # In thông báo đang tính toán chỉ số đánh giá
        print("[CLUSTERING] Đang tính toán Silhouette Score và Davies-Bouldin Index...")
        # Kiểm tra nếu kích thước dữ liệu lớn thì lấy mẫu ngẫu nhiên để tối ưu thời gian
        if len(X) > sample_size:
            # Khởi tạo bộ sinh số ngẫu nhiên với seed cố định
            rng = np.random.RandomState(self.random_state)
            # Chọn ngẫu nhiên các chỉ số mẫu
            sample_idx = rng.choice(len(X), size=sample_size, replace=False)
            # Lấy tập dữ liệu con theo chỉ số mẫu
            X_eval = X[sample_idx]
            # Lấy nhãn cụm tương ứng theo chỉ số mẫu
            labels_eval = self.labels_[sample_idx]
        # Trường hợp dữ liệu nhỏ hơn sample_size thì sử dụng toàn bộ
        else:
            # Sử dụng toàn bộ ma trận đặc trưng
            X_eval = X
            # Sử dụng toàn bộ mảng nhãn cụm
            labels_eval = self.labels_
        # Tính toán hệ số Silhouette Score đo độ phân tách giữa các cụm
        sil_score = float(silhouette_score(X_eval, labels_eval))
        # Tính toán chỉ số Davies-Bouldin Index đánh giá độ đặc và tách biệt cụm
        db_index = float(davies_bouldin_score(X_eval, labels_eval))
        # In kết quả chỉ số đánh giá ra màn hình với độ chính xác 4 chữ số thập phân
        print(f"[CLUSTERING] -> Silhouette Score: {sil_score:.4f} | Davies-Bouldin Index: {db_index:.4f}")
        # Trả về từ điển chứa hai chỉ số đo lường làm tròn 4 chữ số thập phân
        return {"Silhouette_Score": round(sil_score, 4), "Davies_Bouldin_Index": round(db_index, 4)}

    # Hàm giảm chiều dữ liệu xuống 2 chiều bằng PCA
    def fit_transform_pca(self, X: np.ndarray) -> np.ndarray:
        # In thông báo bắt đầu giảm chiều dữ liệu bằng PCA
        print("[CLUSTERING] Chiếu dữ liệu xuống không gian PCA 2D...")
        # Thực hiện giảm chiều không gian dữ liệu xuống 2 thành phần chính
        pca_2d = self.pca.fit_transform(X)
        # Tính tổng tỷ lệ phương sai được giải thích bởi 2 thành phần PCA
        explained_variance = float(np.sum(self.pca.explained_variance_ratio_) * 100)
        # In thông tin phương sai ra màn hình console
        print(f"[CLUSTERING] 2 thành phần PCA giải thích được {explained_variance:.2f}% phương sai dữ liệu.")
        # Trả về ma trận 2 chiều sau khi giảm chiều
        return pca_2d

    # Hàm tính toán các giá trị thống kê trung bình để mô tả chân dung khách hàng của 4 cụm
    def summarize_personas(self, df: pd.DataFrame, cluster_labels: np.ndarray) -> pd.DataFrame:
        # In thông báo bắt đầu tổng hợp thống kê chân dung cụm
        print("[CLUSTERING] Đang tổng hợp số liệu thống kê persona cho từng cụm...")
        # Tạo bản sao DataFrame để không làm thay đổi dữ liệu gốc
        data_with_clusters = df.copy()
        # Gán nhãn cụm vào bảng dữ liệu
        data_with_clusters['Cluster'] = cluster_labels
        # Danh sách các cột chỉ số cần tính toán giá trị trung bình
        numeric_cols = [
            # Chi tiêu trung bình hàng tháng
            'Monthly_Spending',
            # Tỷ lệ sử dụng hạn mức tín dụng trung bình
            'Credit_Utilization',
            # Điểm tín dụng trung bình
            'Credit_Score',
            # Tỷ lệ thanh toán nợ trung bình
            'Payment_Ratio',
            # Tỷ lệ chi tiêu danh mục mua sắm trực tuyến
            'Pct_Online',
            # Tỷ lệ chi tiêu danh mục tạp hóa thực phẩm
            'Pct_Grocery',
            # Tỷ lệ chi tiêu danh mục nhiên liệu xăng dầu
            'Pct_Fuel',
            # Tỷ lệ chi tiêu danh mục nhà hàng ăn uống
            'Pct_Dining',
            # Tỷ lệ chi tiêu danh mục du lịch
            'Pct_Travel',
            # Tỷ lệ chi tiêu danh mục giải trí
            'Pct_Entertainment',
            # Tỷ lệ chi tiêu danh mục hóa đơn tiện ích
            'Pct_Utility',
            # Tỷ lệ chi tiêu hưởng thụ
            'Discretionary_Ratio',
            # Tỷ lệ chi tiêu thiết yếu
            'Essential_Ratio'
        # Kết thúc danh sách cột tính trung bình
        ]
        # Tính giá trị trung bình của các cột chỉ số theo từng cụm
        cluster_summary = data_with_clusters.groupby('Cluster')[numeric_cols].mean()
        # Đếm số lượng khách hàng trong mỗi cụm
        cluster_counts = data_with_clusters['Cluster'].value_counts().sort_index()
        # Chèn cột số lượng khách hàng vào vị trí đầu tiên của bảng
        cluster_summary.insert(0, 'Customer_Count', cluster_counts)
        # Tính tỷ lệ phần trăm số lượng khách hàng của từng cụm
        cluster_summary.insert(1, 'Customer_Percentage', (cluster_counts / len(data_with_clusters)) * 100.0)
        # Làm tròn tất cả các giá trị trong bảng đến 2 chữ số thập phân
        cluster_summary = cluster_summary.round(2)
        # In thông báo đã hoàn thành tính toán thống kê
        print("[CLUSTERING] Tổng hợp số liệu persona 4 cụm hoàn tất!")
        # Trả về bảng thống kê persona
        return cluster_summary
