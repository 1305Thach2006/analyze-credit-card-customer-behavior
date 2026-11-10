# Import thư viện cảnh báo để ẩn các thông báo nâng cấp phiên bản trong tương lai của scikit-learn
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Import thư viện pandas để xử lý cấu trúc dữ liệu bảng
import pandas as pd
# Import thư viện numpy để thực hiện tính toán ma trận và mảng
import numpy as np
# Import các mô hình Uplift Meta-Learners từ thư viện scikit-uplift
from sklift.models import SoloModel, TwoModels
# Import các thước đo kiểm định hiệu năng Qini Score từ scikit-uplift
from sklift.metrics import qini_auc_score, uplift_at_k
# Import bộ phân loại LightGBM để làm mô hình nền tảng (Base Learner)
from lightgbm import LGBMClassifier
# Import hàm chia tập dữ liệu train/test từ scikit-learn
from sklearn.model_selection import train_test_split
# Import typing để chú thích kiểu dữ liệu rõ ràng
from typing import Dict, Any, List, Tuple

# Lớp điều phối toàn diện mô hình học máy nhân quả Uplift Modeling
class UpliftPersonalizationEngine:
    # Hàm khởi tạo lớp UpliftPersonalizationEngine với các tham số cấu hình
    def __init__(self, random_state: int = 42):
        # Cố định seed ngẫu nhiên để đảm bảo tính tái lập kết quả
        self.random_state = random_state
        # Khởi tạo mô hình S-Learner (SoloModel) trên nền tảng LightGBM
        self.s_learner = SoloModel(
            LGBMClassifier(
                n_estimators=60,
                learning_rate=0.05,
                max_depth=5,
                random_state=random_state,
                verbose=-1
            )
        )
        # Khởi tạo mô hình T-Learner (TwoModels) độc lập cho 2 nhóm Treatment và Control
        self.t_learner = TwoModels(
            estimator_trmnt=LGBMClassifier(
                n_estimators=60,
                learning_rate=0.05,
                max_depth=5,
                random_state=random_state,
                verbose=-1
            ),
            estimator_ctrl=LGBMClassifier(
                n_estimators=60,
                learning_rate=0.05,
                max_depth=5,
                random_state=random_state,
                verbose=-1
            ),
            method='vanilla'
        )
        # Biến cờ lưu trạng thái mô hình đã được huấn luyện hay chưa
        self.is_fitted = False
        # Lưu trữ kết quả đối chuẩn hiệu năng các mô hình
        self.metrics_summary = {}

    # Hàm tạo lập biến can thiệp Treatment (W) và biến kết quả chuyển đổi Outcome (Y)
    def prepare_uplift_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        # In thông báo tạo biến can thiệp và biến mục tiêu
        print("[UPLIFT] Đang thiết lập biến can thiệp chiến dịch ưu đãi (Treatment) và biến mục tiêu (Outcome)...")
        # Trích xuất các biến tài chính cốt lõi phục vụ bài toán nhân quả
        score = df["Credit_Score"].values
        util = df["Credit_Utilization"].values
        pay = df["Payment_Ratio"].values
        spend = df["Monthly_Spending"].values

        # Cố định seed sinh số ngẫu nhiên
        np.random.seed(self.random_state)

        # Tính xác suất can thiệp ưu đãi dựa trên điểm tín dụng và mức độ tương tác app
        propensity = 1.0 / (1.0 + np.exp(-((score - 600.0) / 60.0) + (util - 0.5) * 1.5))
        propensity = np.clip(propensity, 0.15, 0.85)
        # Sinh biến can thiệp nhị phân Treatment (W in {0, 1}): 1 là nhận gói ưu đãi, 0 là nhóm đối chứng
        treatment = np.random.binomial(1, propensity)

        # Mô hình hóa hiệu ứng gia tăng thực tế (Treatment Effect tau(X)):
        # Nhóm khách hàng có điểm tín dụng tốt, không kịch trần hạn mức sẽ phản hồi tích cực với ưu đãi
        tau_effect = 0.32 * (score >= 600) * (util < 0.70) * (pay >= 0.45) - 0.12 * (util >= 0.75)
        # Xác suất chuyển đổi cơ sở (Base rate khi không có can thiệp)
        base_rate = 0.22 + 0.28 * (score >= 650) * (spend >= 20000)
        # Tổng xác suất chuyển đổi khi có hoặc không có can thiệp
        prob_outcome = np.clip(base_rate + tau_effect * treatment, 0.05, 0.95)
        # Sinh biến kết quả chuyển đổi Outcome (Y in {0, 1})
        outcome = np.random.binomial(1, prob_outcome)

        # In thông tin phân bổ
        print(f"[UPLIFT] Tỷ lệ nhóm nhận ưu đãi (Treatment = 1): {treatment.mean() * 100:.4f}% | Nhóm đối chứng (Control = 0): {(1 - treatment.mean()) * 100:.4f}%")
        print(f"[UPLIFT] Tỷ lệ khách hàng chuyển đổi thành công (Outcome = 1): {outcome.mean() * 100:.4f}%")
        # Trả về cặp mảng Treatment và Outcome
        return treatment, outcome

    # Hàm huấn luyện và đối chuẩn 2 mô hình T-Learner và S-Learner
    def fit_and_evaluate(self, X: np.ndarray, y: np.ndarray, treatment: np.ndarray) -> Dict[str, Any]:
        # In thông báo bắt đầu huấn luyện Uplift
        print("[UPLIFT] Đang chia tách 80% Train / 20% Test để huấn luyện đối chuẩn T-Learner vs S-Learner...")
        # Chia dữ liệu huấn luyện và kiểm thử
        X_train, X_test, y_train, y_test, trt_train, trt_test = train_test_split(
            X, y, treatment, test_size=0.2, random_state=self.random_state, stratify=treatment
        )

        # Huấn luyện mô hình T-Learner (TwoModels)
        print("[UPLIFT] Đang huấn luyện mô hình T-Learner (Hai nhánh độc lập Treatment và Control)...")
        self.t_learner.fit(X_train, y_train, trt_train)
        # Dự báo điểm Uplift Score của T-Learner trên tập kiểm thử
        uplift_preds_t = self.t_learner.predict(X_test)

        # Huấn luyện mô hình S-Learner (SoloModel)
        print("[UPLIFT] Đang huấn luyện mô hình S-Learner (Mô hình đơn kết hợp biến can thiệp)...")
        self.s_learner.fit(X_train, y_train, trt_train)
        # Dự báo điểm Uplift Score của S-Learner trên tập kiểm thử
        uplift_preds_s = self.s_learner.predict(X_test)

        # Đánh giá hệ số Qini AUC Score cho T-Learner
        qini_score_t = qini_auc_score(y_test, uplift_preds_t, trt_test)
        # Đánh giá hệ số Qini AUC Score cho S-Learner
        qini_score_s = qini_auc_score(y_test, uplift_preds_s, trt_test)

        # Đánh giá tỷ lệ Uplift tại top 30% khách hàng tiềm năng nhất
        uplift_top30_t = uplift_at_k(y_test, uplift_preds_t, trt_test, strategy='overall', k=0.3)
        uplift_top30_s = uplift_at_k(y_test, uplift_preds_s, trt_test, strategy='overall', k=0.3)

        # In kết quả đánh giá đối chuẩn
        print(f"[UPLIFT] Kết quả T-Learner: Qini Score = {qini_score_t:.4f} | Uplift@30% = {uplift_top30_t:.4f}")
        print(f"[UPLIFT] Kết quả S-Learner: Qini Score = {qini_score_s:.4f} | Uplift@30% = {uplift_top30_s:.4f}")

        # Đánh dấu mô hình đã huấn luyện thành công
        self.is_fitted = True
        # Lưu kết quả kiểm định vào từ điển làm tròn 4 chữ số thập phân
        self.metrics_summary = {
            'qini_score_t_learner': round(float(qini_score_t), 4),
            'qini_score_s_learner': round(float(qini_score_s), 4),
            'uplift_top30_t_learner': round(float(uplift_top30_t), 4),
            'uplift_top30_s_learner': round(float(uplift_top30_s), 4),
            'X_test': X_test,
            'y_test': y_test,
            'treatment_test': trt_test,
            'uplift_preds_t': uplift_preds_t,
            'uplift_preds_s': uplift_preds_s
        }
        # Trả về kết quả đánh giá
        return self.metrics_summary

    # Hàm dự báo điểm Uplift Score cho toàn bộ tập dữ liệu bằng mô hình tối ưu T-Learner
    def predict_full_dataset(self, X: np.ndarray) -> np.ndarray:
        # Kiểm tra nếu mô hình chưa huấn luyện thì báo lỗi
        if not self.is_fitted:
            raise RuntimeError("Mô hình Uplift chưa được huấn luyện! Vui lòng gọi fit_and_evaluate trước.")
        # Dự báo điểm Uplift Score cho toàn bộ khách hàng bằng T-Learner
        full_uplift_scores = self.t_learner.predict(X)
        # Làm tròn 4 chữ số thập phân
        return np.round(full_uplift_scores, 4)

    # Hàm phân loại 50.000 khách hàng thành 4 nhóm nhân quả (Uplift Quadrants) chuẩn tiếng Anh
    def assign_uplift_quadrants(self, df: pd.DataFrame, uplift_scores: np.ndarray) -> pd.Series:
        # In thông báo phân loại khách hàng
        print("[UPLIFT] Đang phân loại khách hàng vào 4 nhóm nhân quả kinh điển (Uplift Quadrants)...")
        # Tính ngưỡng phân vị 60% của điểm Uplift để nhận diện nhóm phản hồi nhạy bén
        uplift_threshold = np.percentile(uplift_scores, 60)

        # Định nghĩa các điều kiện gán nhóm nhân quả
        # 1. Nhóm Lost Causes: Khách hàng có rủi ro cao (vi phạm hạn mức, điểm tín dụng kém, trả nợ kém)
        cond_lost_causes = (df["Risk_Flag"] == 1) | (df["Payment_Ratio"] < 0.40) | (df["Credit_Score"] < 550)
        # 2. Nhóm Persuadables: Không bị rủi ro và có Uplift Score vượt ngưỡng (có ưu đãi thì tăng vọt chi tiêu)
        cond_persuadables = (~cond_lost_causes) & (uplift_scores >= uplift_threshold)
        # 3. Nhóm Sure Things: Không rủi ro, tự thân chi tiêu cao (>25.000 VNĐ) hoặc đã đổi điểm thưởng dù Uplift thấp
        cond_sure_things = (~cond_lost_causes) & (~cond_persuadables) & ((df["Monthly_Spending"] >= 25000) | (df["Reward_Points_Redeemed"] > 0))
        # 4. Nhóm Sleeping Dogs: Các khách hàng còn lại có Uplift Score âm hoặc thấp nhạy cảm
        # Danh sách các điều kiện theo thứ tự ưu tiên
        conditions = [cond_lost_causes, cond_persuadables, cond_sure_things]
        # Danh sách tên 4 nhóm nhân quả chuẩn tiếng Anh
        choices = [
            "Lost Causes",
            "Persuadables",
            "Sure Things"
        ]
        # Sử dụng np.select để gán nhãn, mặc định là Sleeping Dogs
        segments = np.select(conditions, choices, default="Sleeping Dogs")
        # In tỷ lệ phân bổ của 4 nhóm nhân quả
        seg_series = pd.Series(segments, index=df.index, name="Uplift_Segment")
        print("[UPLIFT] Tỷ lệ phân bổ 4 nhóm nhân quả (Uplift Quadrants):")
        for seg, count in seg_series.value_counts().items():
            print(f"  • {seg}: {count} khách hàng ({count / len(df) * 100:.4f}%)")
        # Trả về Series phân loại nhóm nhân quả
        return seg_series

    # Hàm tổng hợp bảng báo cáo định lượng đặc tính 4 nhóm nhân quả với tiêu đề tiếng Anh và làm tròn 4 chữ số thập phân
    def generate_uplift_summary_report(self, df: pd.DataFrame, uplift_scores: np.ndarray, segments: pd.Series) -> pd.DataFrame:
        # In thông báo tổng hợp báo cáo định lượng
        print("[UPLIFT] Đang tổng hợp báo cáo định lượng 4 nhóm nhân quả Uplift...")
        # Tạo bảng tạm thời chứa các thông tin cốt lõi
        temp_df = df.copy()
        temp_df["Uplift_Score"] = uplift_scores
        temp_df["Uplift_Segment"] = segments

        # Nhóm theo phân khúc Uplift và tính các giá trị trung bình định lượng (làm tròn 4 số thập phân)
        summary = temp_df.groupby("Uplift_Segment").agg(
            Customer_Count=("Customer_ID", "count"),
            Distribution_Pct=("Customer_ID", lambda x: round(len(x) / len(temp_df) * 100, 4)),
            Mean_Uplift_Score=("Uplift_Score", lambda x: round(x.mean(), 4)),
            Mean_Monthly_Spending=("Monthly_Spending", lambda x: round(x.mean(), 4)),
            Mean_Credit_Limit=("Credit_Limit", lambda x: round(x.mean(), 4)),
            Mean_Credit_Score=("Credit_Score", lambda x: round(x.mean(), 4)),
            Mean_Credit_Utilization=("Credit_Utilization", lambda x: round(x.mean() * 100, 4)),
            Mean_Payment_Ratio=("Payment_Ratio", lambda x: round(x.mean() * 100, 4))
        ).reset_index()

        # In thông báo hoàn tất
        print("[UPLIFT] Hoàn tất tổng hợp báo cáo định lượng 4 nhóm nhân quả!")
        # Trả về DataFrame báo cáo tổng hợp
        return summary
