# Import thu vien pandas de xu ly cac cau truc du lieu bang
import pandas as pd
# Import thu vien numpy de thuc hien cac phep toan ma tran va dieu kien
import numpy as np
# Import thu vien NearestNeighbors tu scikit-learn de tim kiem lang gieng tuong dong
from sklearn.neighbors import NearestNeighbors
# Import thu vien typing de chu thich kieu du lieu
from typing import Dict, Any, List

# Lop tinh diem thuong va goi y uu dai dua tren hanh vi khach hang
class CustomerBehaviorScorer:
    # Ham khoi tao lop CustomerBehaviorScorer voi cac bang tra he so va nguong
    def __init__(self, base_multipliers: Dict[str, float], expense_bonuses: Dict[str, float], risk_thresholds: Dict[str, float]):
        # Luu tru bang he so tinh diem thuong theo hang the tin dung
        self.base_multipliers = base_multipliers
        # Luu tru bang he so thuong theo danh muc chi tieu chu dao
        self.expense_bonuses = expense_bonuses
        # Luu tru cac nguong chi so an toan tai chinh
        self.risk_thresholds = risk_thresholds
        # Khoi tao tu dien de luu tru mo hinh k-NN rieng cho tung cum
        self.cluster_knn_models = {}

    # Ham tinh toan diem thuong dong da bien dua tren hanh vi chi tieu va hang the
    def calculate_dynamic_reward_points(self, df: pd.DataFrame) -> pd.Series:
        # In thong bao bat dau tinh toan diem thuong
        print("[SCORING] Dang tinh toan diem thuong dong dua tren he so hang the va chi tieu...")
        # Tinh he so diem thuong co ban dua vao hang the bang cach map qua tu dien
        base_factor = df["Card_Type"].map(self.base_multipliers).fillna(1.0)
        # Tinh he so thuong danh muc dua vao danh muc chi tieu co gia tri lon nhat
        category_bonus = df["Dominant_Expense_Category"].map(self.expense_bonuses).fillna(0.5)
        # Tinh diem thuong khuyen khich cho khach hang co muc do tuong tac ung dung di dong cao
        app_engagement_bonus = np.where(df["Mobile_App_Login"] >= 10, 0.2, np.where(df["Mobile_App_Login"] >= 5, 0.1, 0.0))
        # Tinh he so giam tru diem thuong doi voi khach hang co ty le thanh toan no thap
        late_payment_penalty = np.where(df["Payment_Ratio"] < self.risk_thresholds["min_payment_ratio"], 0.5, 0.0)
        # Tinh tong he so thuong cuoi cung sau khi cong thuong va tru phat
        total_multiplier = (base_factor + category_bonus + app_engagement_bonus - late_payment_penalty).clip(lower=0.5)
        # Tinh so diem thuong dong ty le thuan voi so tien chi tieu hang thang
        calculated_points = (df["Monthly_Spending"] / 500.0) * total_multiplier
        # Lam tron diem thuong den chu so thap phan thu nhat
        reward_points = calculated_points.round(1)
        # In thong bao da hoan thanh tinh toan diem thuong
        print(f"[SCORING] Hoan tat tinh diem thuong! Gia tri trung binh toan bo du lieu: {reward_points.mean():.1f} diem.")
        # Tra ve chuoi diem thuong dong cua tat ca khach hang
        return reward_points

    # Ham khoi tao va huan luyen mo hinh k-NN doc lap cho tung cum khach hang
    def fit_knn_by_cluster(self, df: pd.DataFrame, feature_cols: List[str], cluster_col: str = 'Cluster') -> None:
        # In thong bao huan luyen k-NN tim lang gieng
        print("[SCORING] Huan luyen thuat toan k-NN tim lang gieng tuong dong theo cum...")
        # Lap qua tung cum trong tap du lieu
        for c in df[cluster_col].unique():
            # Loc tap du lieu con thuoc cum
            cluster_subset = df[df[cluster_col] == c]
            # Khoi tao mo hinh NearestNeighbors voi k=5
            knn = NearestNeighbors(n_neighbors=min(5, len(cluster_subset)), metric='euclidean')
            # Huan luyen mo hinh k-NN tren tap dac trung cua cum
            knn.fit(cluster_subset[feature_cols].values)
            # Luu tru mo hinh va danh sach ma khach hang tuong ung vao tu dien
            self.cluster_knn_models[c] = {'model': knn, 'customer_ids': cluster_subset['Customer_ID'].values, 'indices': cluster_subset.index.values}
        # In thong bao da hoan thanh huan luyen k-NN
        print("[SCORING] Khoi tao thuat toan k-NN hoan tat!")

    # Ham kiem tra dieu kien rui ro tin dung dua tren cac nguong chi so
    def check_risk_status(self, customer_row: pd.Series) -> str:
        # Kiem tra neu ty le su dung han muc vuot 75%
        if customer_row["Credit_Utilization"] > self.risk_thresholds["max_credit_utilization"]:
            # Tra ve trang thai canh bao rui ro cao
            return "HIGH_RISK_UTILIZATION"
        # Kiem tra neu diem tin dung duoi 550
        if customer_row["Credit_Score"] < self.risk_thresholds["min_credit_score"]:
            # Tra ve trang thai canh bao diem tin dung thap
            return "HIGH_RISK_CREDIT_SCORE"
        # Kiem tra neu ty le thanh toan duoi 40%
        if customer_row["Payment_Ratio"] < self.risk_thresholds["min_payment_ratio"]:
            # Tra ve trang thai canh bao cham tra
            return "HIGH_RISK_PAYMENT_RATIO"
        # Tra ve trang thai an toan neu vuot qua tat ca cac kiem tra
        return "SAFE"

    # Ham tong hop bao cao phan tich chi tiet cho tung khach hang tren toan bo tap du lieu
    def generate_customer_report(self, df: pd.DataFrame) -> pd.DataFrame:
        # In thong bao tao bao cao khach hang
        print("[REPORT] Dang tong hop bao cao phan tich chi tiet cho toan bo tung khach hang...")
        # Tao danh sach cac dieu kien kiem tra rui ro tin dung
        conditions = [df["Credit_Utilization"] > self.risk_thresholds["max_credit_utilization"], df["Credit_Score"] < self.risk_thresholds["min_credit_score"], df["Payment_Ratio"] < self.risk_thresholds["min_payment_ratio"]]
        # Tao danh sach cac nhan rui ro tuong ung theo thu tu uu tien
        choices = ["HIGH_RISK_UTILIZATION", "HIGH_RISK_CREDIT_SCORE", "HIGH_RISK_PAYMENT_RATIO"]
        # Gan nhan trang thai rui ro su dung phep chon theo dieu kien cua numpy
        risk_status = np.select(conditions, choices, default="SAFE")
        # Danh sach cac cot can trinh bay trong bao cao chi tiet
        report_columns = [
            # Ma dinh danh khach hang duy nhat
            "Customer_ID",
            # Nhan cum phan loai hanh vi
            "Cluster",
            # Hang the tin dung hien tai
            "Card_Type",
            # Tong so tien chi tieu hang thang
            "Monthly_Spending",
            # Danh muc chi tieu co gia tri lon nhat
            "Dominant_Expense_Category",
            # Han muc tin dung duoc ngan hang cap
            "Credit_Limit",
            # Ty le su dung han muc tin dung
            "Credit_Utilization",
            # Diem uy tin tin dung
            "Credit_Score",
            # Ty le thanh toan du no hang thang
            "Payment_Ratio",
            # Diem thuong dong da duoc tinh toan
            "Dynamic_Reward_Points",
            # Co nhi phan canh bao rui ro
            "Risk_Flag",
            # Ty trong chi tieu cho cac danh muc tuy y
            "Discretionary_Ratio",
            # Ty trong chi tieu cho cac danh muc thiet yeu
            "Essential_Ratio",
            # Ty le doi diem thuong
            "Redemption_Rate"
        # Ket thuc khoi danh sach hoac tu dien
        ]
        # Tao DataFrame bao cao chua cac cot da chon
        report_df = df[report_columns].copy()
        # Them cot trang thai rui ro chi tiet
        report_df["Risk_Status"] = risk_status
        # In thong bao hoan tat bao cao
        print(f"[REPORT] Da tong hop thanh cong bao cao cho toan bo {len(report_df)} khach hang!")
        # Tra ve bang bao cao chi tiet
        return report_df

    # Ham phan tich chuyen sau cho mot khach hang cu the
    def analyze_customer(self, customer_id: str, df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
        # Loc thong tin khach hang theo ma dinh danh Customer_ID
        cust_match = df[df['Customer_ID'] == customer_id]
        # Kiem tra neu khong tim thay khach hang
        if cust_match.empty:
            # Tra ve thong bao loi
            return {"error": f"Khong tim thay ma khach hang: {customer_id}"}
        # Lay ban ghi thong tin khach hang
        customer_row = cust_match.iloc[0]
        # Lay nhan cum cua khach hang
        cluster_id = customer_row['Cluster']
        # Kiem tra trang thai rui ro tin dung
        risk_status = self.check_risk_status(customer_row)
        # Danh sach ma khach hang lang gieng tuong dong
        neighbor_ids = []
        # Kiem tra neu da huan luyen mo hinh k-NN cho cum nay
        if cluster_id in self.cluster_knn_models:
            # Lay thong tin mo hinh k-NN cua cum
            knn_info = self.cluster_knn_models[cluster_id]
            # Lay vector dac trung cua khach hang
            cust_vec = customer_row[feature_cols].values.reshape(1, -1)
            # Tim kiem 5 lang gieng gan nhat
            distances, indices = knn_info['model'].kneighbors(cust_vec)
            # Lay danh sach ma khach hang tuong ung voi cac chi so tim duoc
            neighbor_ids = list(knn_info['customer_ids'][indices[0]])
            # Loai bo chinh ma khach hang hien tai neu co trong danh sach lang gieng
            if customer_id in neighbor_ids:
                # Xoa ma khach hang khoi danh sach lang gieng
                neighbor_ids.remove(customer_id)
        # Tao ho so phan tich tong hop cua khach hang
        profile = {
            # Ma khach hang
            "Customer_ID": customer_id,
            # Nhan cum
            "Cluster": int(cluster_id),
            # Hang the tin dung
            "Card_Type": customer_row["Card_Type"],
            # Chi tieu hang thang
            "Monthly_Spending": float(customer_row["Monthly_Spending"]),
            # Danh muc chu dao
            "Dominant_Expense_Category": customer_row["Dominant_Expense_Category"],
            # Diem thuong dong
            "Dynamic_Reward_Points": float(customer_row["Dynamic_Reward_Points"]),
            # Trang thai rui ro
            "Risk_Status": risk_status,
            # Co rui ro nhi phan
            "Risk_Flag": int(customer_row["Risk_Flag"]),
            # Danh sach 5 lang gieng gan nhat
            "Similar_Neighbors": neighbor_ids[:5]
        # Ket thuc khoi danh sach hoac tu dien
        }
        # Tra ve ho so khach hang
        return profile
