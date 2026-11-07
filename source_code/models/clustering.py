# Import thu vien KMeans tu scikit-learn de phan cum du lieu
from sklearn.cluster import KMeans
# Import thu vien PCA tu scikit-learn de giam chieu du lieu
from sklearn.decomposition import PCA
# Import thu vien tinh chi so do luong phan tach Silhouette Score
from sklearn.metrics import silhouette_score
# Import thu vien tinh chi so danh gia do dac cum Davies-Bouldin Index
from sklearn.metrics import davies_bouldin_score
# Import thu vien numpy de tinh toan ma tran va mang so hoc
import numpy as np
# Import thu vien pandas de quan ly bang ket qua thong ke
import pandas as pd
# Import thu vien typing de dinh nghia kieu du lieu
from typing import Dict, Any, Tuple

# Lop thuc thi quy trinh phan cum bang thuat toan K-Means va PCA
class CustomerClustering:
    # Ham khoi tao lop CustomerClustering voi so cum va hat giong ngau nhien
    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        # Luu so luong cum can phan chia
        self.n_clusters = n_clusters
        # Luu gia tri hat giong de co dinh ket qua
        self.random_state = random_state
        # Khoi tao mo hinh KMeans su dung thuat toan k-means++ khoi tao tam cum
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=self.random_state
        )
        # Khoi tao doi tuong PCA giam xuong 2 chieu de truc quan hoa
        self.pca = PCA(n_components=2, random_state=self.random_state)
        # Bien luu tru mang nhan cum cua toan bo tap du lieu
        self.labels_ = None

    # Ham huan luyen va du doan nhan cum cho toan bo khach hang
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        # In thong bao bat dau phan cum K-Means
        print(f"[CLUSTERING] Bat dau phan cum K-Means voi k = {self.n_clusters}...")
        # Thuc hien huan luyen va gan nhan cum cho tap du lieu
        self.labels_ = self.kmeans.fit_predict(X)
        # In thong bao da hoan thanh phan cum
        print("[CLUSTERING] Phan cum K-Means hoan tat!")
        # Tra ve mang chua nhan cum cua cac khach hang
        return self.labels_

    # Ham danh gia chat luong phan cum bang cac chi so khoa hoc
    def evaluate(self, X: np.ndarray, sample_size: int = 10000) -> Dict[str, float]:
        # In thong bao dang tinh toan chi so danh gia
        print("[CLUSTERING] Dang tinh toan Silhouette Score va Davies-Bouldin Index...")
        # Lay mau ngau nhien co dinh 10.000 diem du lieu de tinh Silhouette Score
        indices = np.random.RandomState(self.random_state).choice(len(X), size=min(len(X), sample_size), replace=False)
        # Tinh toan chi so Silhouette Score
        sil_score = float(silhouette_score(X[indices], self.labels_[indices]))
        # Tinh toan chi so Davies-Bouldin Index
        db_score = float(davies_bouldin_score(X, self.labels_))
        # In ket qua chi so danh gia ra man hinh
        print(f"[CLUSTERING] -> Silhouette Score: {sil_score:.4f} | Davies-Bouldin Index: {db_score:.4f}")
        # Tra ve tu dien chua 2 chi so do luong
        return {'silhouette_score': sil_score, 'davies_bouldin_index': db_score}

    # Ham giam chieu du lieu xuong 2D bang PCA de truc quan hoa
    def fit_transform_pca(self, X: np.ndarray) -> np.ndarray:
        # In thong bao thuc hien giam chieu PCA
        print("[CLUSTERING] Chieu du lieu xuong khong gian PCA 2D...")
        # Tinh toan va bien doi tap du lieu thanh 2 thanh phan chinh
        pca_2d = self.pca.fit_transform(X)
        # Tinh tong ty le phuong sai duoc giai thich boi 2 thanh phan chinh
        explained_var = np.sum(self.pca.explained_variance_ratio_) * 100
        # In ty le phuong sai giai thich duoc
        print(f"[CLUSTERING] 2 thanh phan PCA giai thich duoc {explained_var:.2f}% phuong sai du lieu.")
        # Tra ve ma tran toa do 2D
        return pca_2d

    # Ham tong hop cac gia tri trung binh cua tung cum de xuat so lieu thong ke
    def summarize_personas(self, df: pd.DataFrame, cluster_labels: np.ndarray) -> pd.DataFrame:
        # Tao ban sao DataFrame
        temp_df = df.copy()
        # Gan nhan cum vao DataFrame
        temp_df['Cluster'] = cluster_labels
        # Danh sach cac cot chi so can tinh trung binh thong ke
        summary_cols = [
            'Monthly_Spending', 'Annual_Income', 'Credit_Limit', 'Credit_Utilization',
            'Credit_Score', 'Payment_Ratio', 'Redemption_Rate', 'Discretionary_Ratio',
            'Essential_Ratio', 'Mobile_App_Login', 'Risk_Flag'
        ]
        # Tinh gia tri trung binh cua cac dac trung theo tung cum
        cluster_summary = temp_df.groupby('Cluster')[summary_cols].mean().reset_index()
        # Dem so luong khach hang thuoc tung cum
        counts = temp_df['Cluster'].value_counts().sort_index().values
        # Tinh ty le phan tram so khach hang trong tung cum
        percentages = (counts / len(temp_df)) * 100
        # Gan cot so luong khach hang
        cluster_summary['Count'] = counts
        # Gan cot ty le phan tram (lam tron 2 chu so thap phan)
        cluster_summary['Percentage'] = percentages.round(2)
        # Tra ve bang thong ke tom tat cac cum
        return cluster_summary
