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
        self.kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', n_init=10, max_iter=300, random_state=self.random_state)
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
        # Kiem tra neu kich thuoc du lieu lon thi lay mau ngau nhien de toi uu thoi gian
        if len(X) > sample_size:
            # Khoi tao bo sinh so ngau nhien voi seed co dinh
            rng = np.random.RandomState(self.random_state)
            # Chon ngau nhien cac chi so mau
            sample_idx = rng.choice(len(X), size=sample_size, replace=False)
            # Lay tap du lieu con theo chi so mau
            X_eval = X[sample_idx]
            # Lay nhan cum tuong ung theo chi so mau
            labels_eval = self.labels_[sample_idx]
        # Truong hop du lieu nho hon sample_size thi su dung toan bo
        else:
            # Su dung toan bo ma tran dac trung
            X_eval = X
            # Su dung toan bo mang nhan cum
            labels_eval = self.labels_
        # Tinh toan he so Silhouette Score do do phan tach giua cac cum
        sil_score = float(silhouette_score(X_eval, labels_eval))
        # Tinh toan chi so Davies-Bouldin Index danh gia do dac va tach biet cum
        db_index = float(davies_bouldin_score(X_eval, labels_eval))
        # In ket qua chi so danh gia ra man hinh
        print(f"[CLUSTERING] -> Silhouette Score: {sil_score:.4f} | Davies-Bouldin Index: {db_index:.4f}")
        # Tra ve tu dien chua hai chi so do luong
        return {"Silhouette_Score": round(sil_score, 4), "Davies_Bouldin_Index": round(db_index, 4)}

    # Ham giam chieu du lieu xuong 2 chieu bang PCA
    def fit_transform_pca(self, X: np.ndarray) -> np.ndarray:
        # In thong bao bat dau giam chieu du lieu bang PCA
        print("[CLUSTERING] Chieu du lieu xuong khong gian PCA 2D...")
        # Thuc hien giam chieu khong gian du lieu xuong 2 thanh phan chinh
        pca_2d = self.pca.fit_transform(X)
        # Tinh tong ty le phuong sai duoc giai thich boi 2 thanh phan PCA
        explained_variance = float(np.sum(self.pca.explained_variance_ratio_) * 100)
        # In thong tin phuong sai ra man hinh console
        print(f"[CLUSTERING] 2 thanh phan PCA giai thich duoc {explained_variance:.2f}% phuong sai du lieu.")
        # Tra ve ma tran 2 chieu sau khi giam chieu
        return pca_2d

    # Ham tong hop gia tri trung binh cua cac dac trung theo tung cum de xay dung persona
    def summarize_personas(self, df: pd.DataFrame, cluster_labels: np.ndarray) -> pd.DataFrame:
        # Tao ban sao DataFrame de tranh anh huong den du lieu goc
        df_copy = df.copy()
        # Gan nhan cum vao cot moi co ten Cluster
        df_copy["Cluster"] = cluster_labels
        # Danh sach cac dac trung quan trong can tong hop theo cum
        summary_features = [
            # Cot tong chi tieu hang thang
            "Monthly_Spending",
            # Cot han muc tin dung duoc cap
            "Credit_Limit",
            # Cot ty le su dung han muc tin dung
            "Credit_Utilization",
            # Cot diem tin dung cua khach hang
            "Credit_Score",
            # Cot ty le thanh toan no hang thang
            "Payment_Ratio",
            # Cot ty trong chi tieu mua sam online
            "Pct_Online",
            # Cot ty trong chi tieu tap hoa
            "Pct_Grocery",
            # Cot ty trong chi tieu nhien lieu
            "Pct_Fuel",
            # Cot ty trong chi tieu an uong nha hang
            "Pct_Dining",
            # Cot ty trong chi tieu du lich
            "Pct_Travel",
            # Cot ty trong chi tieu giai tri
            "Pct_Entertainment",
            # Cot ty trong chi tieu hoa don sinh hoat
            "Pct_Utility",
            # Cot ty le doi diem thuong
            "Redemption_Rate",
            # Cot co canh bao rui ro tin dung
            "Risk_Flag"
        # Ket thuc khoi danh sach hoac tu dien
        ]
        # Tinh gia tri trung binh cho tung dac trung nhom theo cum
        summary = df_copy.groupby("Cluster")[summary_features].mean().round(2)
        # Tinh so luong khach hang thuoc tung cum
        summary["Customer_Count"] = df_copy.groupby("Cluster").size()
        # Tinh ty le phan tram khach hang cua tung cum tren tong the
        summary["Percentage"] = (summary["Customer_Count"] / len(df_copy) * 100).round(2)
        # Dat ten cho chi muc cum
        summary.index.name = "Cluster"
        # Tra ve bang tong hop cac persona dinh luong
        return summary
