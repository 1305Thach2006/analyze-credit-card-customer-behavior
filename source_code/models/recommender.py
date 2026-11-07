# Import thu vien NearestNeighbors tu scikit-learn de tim kiem lang gieng tuong dong
from sklearn.neighbors import NearestNeighbors
# Import thu vien numpy de tinh toan so hoc va ma tran
import numpy as np
# Import thu vien pandas de thao tac tren DataFrame
import pandas as pd
# Import thu vien typing de chu thich kieu du lieu
from typing import Dict, Any, List

# Lop quan ly thuat toan tinh toan diem thuong dong va phan tich tuong dong hanh vi
class CustomerBehaviorScorer:
    # Ham khoi tao bo phan tich va tinh diem
    def __init__(self, base_multipliers: dict, expense_bonuses: dict, risk_thresholds: dict):
        # Luu tu dien he so hang the
        self.base_multipliers = base_multipliers
        # Luu tu dien he so thuong theo danh muc chi tieu chu dao
        self.expense_bonuses = expense_bonuses
        # Luu cac nguong danh gia rui ro tai chinh
        self.risk_thresholds = risk_thresholds
        # Bo nho luu tru cac mo hinh k-NN theo tung cum
        self.cluster_knn_models = {}

    # Ham tinh toan chi so diem thuong dong dua tren cong thuc da bien
    def calculate_dynamic_reward_points(self, df: pd.DataFrame) -> pd.Series:
        # In thong bao bat dau tinh toan diem thuong
        print("[SCORING] Dang tinh toan diem thuong dong dua tren he so hang the va chi tieu...")
        # Lay he so co ban theo hang the
        rct = df["Card_Type"].map(self.base_multipliers).fillna(0.5)
        # Lay he so thuong them theo danh muc chi tieu co gia tri lon nhat
        et = df["Dominant_Expense_Category"].map(self.expense_bonuses).fillna(0.5)
        # Tinh diem thuong tuong tac so cho khach hang co so lan dang nhap app tren 30 lan
        app_bonus = np.where(df["Mobile_App_Login"] > 30, 1.0, 0.0)
        # Tinh diem phat neu ty le thanh toan duoi nguong an toan
        risk_penalty = np.where(df["Payment_Ratio"] < self.risk_thresholds["min_payment_ratio"], 1.5, 0.0)
        # Tong hop tong he so nhan diem thuong
        scored_points = rct + et + app_bonus - risk_penalty
        # Dam bao he so luon duong
        scored_points = np.maximum(0.1, scored_points)
        # Tinh tong diem thuong quy doi tren moi 500 don vi chi tieu
        reward_points = (df["Monthly_Spending"] / 500.0) * scored_points
        # Lam tron diem thuong thanh so nguyen
        reward_points = reward_points.round().astype(int)
        # In thong bao hoan tat tinh toan diem thuong
        print(f"[SCORING] Hoan tat tinh diem thuong! Gia tri trung binh toan bo du lieu: {reward_points.mean():.1f} diem.")
        # Tra ve chuoi ket qua diem thuong
        return reward_points

    # Ham huan luyen thuat toan k-NN de tim cac khach hang co hanh vi gan nhat theo cum
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
            # Luu tru mo hinh va danh sach ma khach hang tuong ung
            self.cluster_knn_models[c] = {
                'model': knn,
                'customer_ids': cluster_subset['Customer_ID'].values,
                'indices': cluster_subset.index.values
            }
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
        conditions = [
            df["Credit_Utilization"] > self.risk_thresholds["max_credit_utilization"],
            df["Credit_Score"] < self.risk_thresholds["min_credit_score"],
            df["Payment_Ratio"] < self.risk_thresholds["min_payment_ratio"]
        ]
        # Tao danh sach cac nhan rui ro tuong ung theo thu tu uu tien
        choices = [
            "HIGH_RISK_UTILIZATION",
            "HIGH_RISK_CREDIT_SCORE",
            "HIGH_RISK_PAYMENT_RATIO"
        ]
        # Gan nhan trang thai rui ro su dung np.select
        risk_status_series = np.select(conditions, choices, default="SAFE")
        # Danh sach cac cot chi so can dua vao bao cao tong hop cho tung khach hang
        report_columns = [
            "Customer_ID", "Cluster", "Card_Type", "Monthly_Spending",
            "Dominant_Expense_Category", "Credit_Limit", "Credit_Utilization",
            "Credit_Score", "Payment_Ratio", "Dynamic_Reward_Points",
            "Risk_Flag", "Discretionary_Ratio", "Essential_Ratio", "Redemption_Rate"
        ]
        # Trich xuat cac cot bao cao tu DataFrame goc
        report_df = df[report_columns].copy()
        # Chen cot trang thai rui ro chi tiet vao bao cao
        report_df["Risk_Status"] = risk_status_series
        # In thong bao hoan tat tao bao cao
        print(f"[REPORT] Da tong hop thanh cong bao cao cho toan bo {len(report_df)} khach hang!")
        # Tra ve bang bao cao chi tiet tung khach hang
        return report_df

    # Ham trich xuat so lieu phan tich chi tiet va cac khach hang tuong dong cho 1 khach hang
    def analyze_customer(self, customer_id: str, df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
        # Tim dong du lieu cua khach hang theo ma ID
        match = df[df["Customer_ID"] == customer_id]
        # Kiem tra neu khong tim thay
        if len(match) == 0:
            # Nem loi neu khong ton tai khach hang
            raise ValueError(f"Khong tim thay ma khach hang: {customer_id}")
        # Lay dong du lieu cua khach hang
        cust = match.iloc[0]
        # Xac dinh trang thai rui ro
        risk_status = self.check_risk_status(cust)
        # Lay cum hien tai cua khach hang
        cluster_id = cust["Cluster"]
        # Lay mo hinh k-NN cua cum tuong ung
        knn_info = self.cluster_knn_models.get(cluster_id)
        # Danh sach ma khach hang tuong dong
        neighbor_ids = []
        # Neu co mo hinh k-NN thi tim 5 lang gieng gan nhat
        if knn_info:
            # Lay vector dac trung cua khach hang
            feat_vec = cust[feature_cols].values.reshape(1, -1)
            # Tim khoang cach va chi muc cua cac diem gan nhat
            distances, indices = knn_info['model'].kneighbors(feat_vec)
            # Lay danh sach ma khach hang cua cac lang gieng gan nhat
            neighbor_ids = list(knn_info['customer_ids'][indices[0]])
        # Tra ve tu dien chua toan bo so lieu phan tich dinh luong
        return {
            "Customer_ID": customer_id,
            "Cluster": int(cluster_id),
            "Dominant_Category": cust["Dominant_Expense_Category"],
            "Monthly_Spending": float(cust["Monthly_Spending"]),
            "Credit_Utilization": float(cust["Credit_Utilization"]),
            "Credit_Score": int(cust["Credit_Score"]),
            "Payment_Ratio": float(cust["Payment_Ratio"]),
            "Calculated_Reward_Points": int(cust.get("Dynamic_Reward_Points", 0)),
            "Risk_Status": risk_status,
            "Nearest_Neighbors": neighbor_ids
        }
